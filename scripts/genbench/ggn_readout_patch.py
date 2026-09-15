"""Add kinematics/GRF read-out to Generative GaitNet's pybind bindings.

Upstream's RayEnvManager exposes only the RL interface (projected states,
actions, rewards) -- nothing that lets you read the actual joint angles or the
ground reaction forces the physics engine computes. Both are needed to put
GaitNet's output in the same schema as the other benchmark models, so this patch
adds five read-only accessors that reach through to the DART skeleton and the
collision solver:

    GetTime()        simulation clock [s]
    GetPositions()   generalised coordinates, all DoFs [rad / m]
    GetVelocities()  generalised velocities
    GetDofNames()    DoF names, in getPositions() order
    GetBodyCOM(name) COM of a named body node [m]
    GetGRF()         [Fx,Fy,Fz]_right, [Fx,Fy,Fz]_left ground reaction force [N]

GRF is summed from the last collision result, attributing each contact to the
right or left foot by walking up the body-node chain to a Talus/Foot/Heel node.

Run this from the GenerativeGaitNet checkout root, then rebuild.
"""
import os
import re
import sys

HEADER_ADDITIONS = r'''
	// ---- benchmark read-out (added: upstream exposes no kinematics/GRF) ----
	double GetTime() { return mEnv->GetWorld()->getTime(); }
	py::array_t<float> GetPositions()
	{
		Eigen::VectorXd q = mEnv->GetCharacter()->GetSkeleton()->getPositions();
		return toNumPyArray(q);
	}
	py::array_t<float> GetVelocities()
	{
		Eigen::VectorXd dq = mEnv->GetCharacter()->GetSkeleton()->getVelocities();
		return toNumPyArray(dq);
	}
	py::list GetDofNames()
	{
		py::list names;
		auto skel = mEnv->GetCharacter()->GetSkeleton();
		for (std::size_t i = 0; i < skel->getNumDofs(); i++)
			names.append(py::str(skel->getDof(i)->getName()));
		return names;
	}
	py::array_t<float> GetBodyCOM(std::string name)
	{
		Eigen::VectorXd com = mEnv->GetCharacter()->GetSkeleton()->getBodyNode(name)->getCOM();
		return toNumPyArray(com);
	}
	py::array_t<float> GetGRF();
	double GetBodyMass() { return mEnv->GetCharacter()->GetSkeleton()->getMass(); }
'''

SOURCE_ADDITIONS = r'''
// Sum contact forces on each foot. DART reports one force per contact point;
// contacts are attributed to a foot by walking up the parent chain until a node
// whose name marks it as part of the right or left foot is found.
// Resolve the BodyNode behind a collision object, or null when the object is
// not attached to one (the ground plane is a SimpleFrame).
static const dart::dynamics::BodyNode *bodyNodeOf(const dart::collision::CollisionObject *obj)
{
	if (obj == nullptr)
		return nullptr;
	const dart::dynamics::ShapeFrame *sf = obj->getShapeFrame();
	if (sf == nullptr)
		return nullptr;
	const dart::dynamics::ShapeNode *sn = sf->asShapeNode();
	if (sn == nullptr)
		return nullptr;
	return sn->getBodyNodePtr().get();
}

static int footSideOf(const dart::dynamics::BodyNode *bn)
{
	while (bn != nullptr)
	{
		const std::string &n = bn->getName();
		bool isFoot = n.find("Talus") != std::string::npos ||
					  n.find("Foot") != std::string::npos ||
					  n.find("Heel") != std::string::npos ||
					  n.find("Toe") != std::string::npos;
		if (isFoot)
		{
			if (!n.empty() && n[n.size() - 1] == 'R')
				return 0;
			if (!n.empty() && n[n.size() - 1] == 'L')
				return 1;
		}
		bn = bn->getParentBodyNode();
	}
	return -1;
}

py::array_t<float> RayEnvManager::GetGRF()
{
	Eigen::VectorXd grf = Eigen::VectorXd::Zero(6);
	auto world = mEnv->GetWorld();
	const auto &result = world->getConstraintSolver()->getLastCollisionResult();
	auto skel = mEnv->GetCharacter()->GetSkeleton();

	for (std::size_t i = 0; i < result.getNumContacts(); i++)
	{
		const auto &c = result.getContact(i);
		// the ground is a SimpleFrame, not a ShapeNode -- asShapeNode() returns
		// null for it, so every hop here has to be guarded.
		auto *bn1 = bodyNodeOf(c.collisionObject1);
		auto *bn2 = bodyNodeOf(c.collisionObject2);

		// force sign convention: c.force acts on body 1
		int side1 = (bn1 && bn1->getSkeleton() == skel) ? footSideOf(bn1) : -1;
		int side2 = (bn2 && bn2->getSkeleton() == skel) ? footSideOf(bn2) : -1;

		if (side1 >= 0)
			grf.segment(3 * side1, 3) += c.force;
		else if (side2 >= 0)
			grf.segment(3 * side2, 3) -= c.force;
	}
	return toNumPyArray(grf);
}
'''

BINDINGS = '''        .def("GetTime", &RayEnvManager::GetTime)
        .def("GetPositions", &RayEnvManager::GetPositions)
        .def("GetVelocities", &RayEnvManager::GetVelocities)
        .def("GetDofNames", &RayEnvManager::GetDofNames)
        .def("GetBodyCOM", &RayEnvManager::GetBodyCOM)
        .def("GetGRF", &RayEnvManager::GetGRF)
        .def("GetBodyMass", &RayEnvManager::GetBodyMass)
'''


def patch_header(path):
    src = open(path).read()
    if 'GetGRF' in src:
        print('header already patched')
        return
    marker = '\tint GetStateDiffNum() { return mEnv->GetStateDiffNum(); }\n'
    if marker not in src:
        sys.exit('header marker not found')
    src = src.replace(marker, marker + HEADER_ADDITIONS)
    open(path, 'w').write(src)
    print('patched', path)


def patch_source(path):
    src = open(path).read()
    if 'RayEnvManager::GetGRF' in src:
        print('source already patched')
        return
    # insert the GRF implementation just before the PYBIND11_MODULE block
    m = re.search(r'PYBIND11_MODULE', src)
    if not m:
        sys.exit('PYBIND11_MODULE not found')
    src = src[:m.start()] + SOURCE_ADDITIONS + '\n' + src[m.start():]

    tail = '        .def("GetStateDiffNum", &RayEnvManager::GetStateDiffNum);'
    if tail not in src:
        sys.exit('binding tail not found')
    src = src.replace(tail, BINDINGS + tail)
    open(path, 'w').write(src)
    print('patched', path)


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    patch_header(os.path.join(root, 'python', 'RayEnvManager.h'))
    patch_source(os.path.join(root, 'python', 'RayEnvManager.cpp'))
