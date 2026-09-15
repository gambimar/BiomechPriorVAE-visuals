import styles from './Selector.module.css';

interface SelectorProps<T extends string> {
  options: { id: T; label: string }[];
  value: T;
  onChange: (id: T) => void;
}

export function Selector<T extends string>({ options, value, onChange }: SelectorProps<T>) {
  return (
    <div className={styles.wrap}>
      {options.map((o) => (
        <button
          key={o.id}
          type="button"
          className={styles.btn}
          data-active={o.id === value}
          onClick={() => onChange(o.id)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
