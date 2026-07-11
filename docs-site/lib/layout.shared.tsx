import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { gitConfig } from './shared';

/** A small teal hexagon mark echoing the APMODE badge. */
function Mark() {
  return (
    <svg
      width="20"
      height="22"
      viewBox="0 0 20 22"
      fill="none"
      aria-hidden
      style={{ flexShrink: 0 }}
    >
      <path
        d="M10 1.2 18.2 6v10L10 20.8 1.8 16V6z"
        stroke="var(--color-fd-primary)"
        strokeWidth="1.4"
        fill="color-mix(in oklab, var(--color-fd-primary) 14%, transparent)"
      />
      <path
        d="M6.2 13.5c1.4-4.4 2.6-6.6 3.8-6.6s2.4 2.2 3.8 6.6"
        stroke="var(--color-fd-primary)"
        strokeWidth="1.4"
        strokeLinecap="round"
        fill="none"
      />
    </svg>
  );
}

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontWeight: 600,
            letterSpacing: '0.02em',
          }}
        >
          <Mark />
          APMODE
        </span>
      ),
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
