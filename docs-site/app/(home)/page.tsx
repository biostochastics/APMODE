import Link from 'next/link';
import Image from 'next/image';

const PARADIGMS = [
  'Classical NLME',
  'Automated search',
  'Bayesian Stan/Torsten',
  'Hybrid mechanistic-NODE',
  'Agentic LLM',
];

const FEATURES = [
  {
    title: 'Formular is the control surface',
    body: 'Every model — classical, Bayesian, or neural — is written in a typed PK DSL, compiled to a validated AST, and lowered to backend-specific code.',
    href: '/docs/guide/formular-dsl',
    cta: 'Explore Formular',
  },
  {
    title: 'Governance is a gated funnel',
    body: 'Gate 1 → 2 → 2.5 → 3 are disqualifying gates, not a weighted sum. Thresholds are versioned policy artifacts, not hard-coded constants.',
    href: '/docs/guide/governance-reproducibility/gates',
    cta: 'See the gates',
  },
  {
    title: 'Reproducibility is the output',
    body: 'Every run emits a sealed JSON bundle — data manifest, search trajectory, gate decisions, lineage DAG — replayable and exportable as an RO-Crate.',
    href: '/docs/guide/governance-reproducibility/reproducibility-bundle',
    cta: 'Inspect a bundle',
  },
  {
    title: 'Three lanes, not one loop',
    body: 'Submission, Discovery, and Optimization are separate pipelines with different admissible backends. NODE and agentic models are never submission-eligible.',
    href: '/docs/guide/concepts/lanes',
    cta: 'Compare lanes',
  },
];

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col">
      {/* ---- Hero: dark instrument panel ---- */}
      <section
        className="relative overflow-hidden border-b border-fd-border"
        style={{
          background:
            'radial-gradient(120% 80% at 50% -10%, hsl(190 60% 20% / 0.55), transparent 60%), linear-gradient(180deg, hsl(200 30% 5%), hsl(200 32% 7%))',
        }}
      >
        {/* faint grid texture */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-[0.14]"
          style={{
            backgroundImage:
              'linear-gradient(hsl(186 60% 70% / 0.6) 1px, transparent 1px), linear-gradient(90deg, hsl(186 60% 70% / 0.6) 1px, transparent 1px)',
            backgroundSize: '44px 44px',
            maskImage:
              'radial-gradient(120% 90% at 50% 0%, black, transparent 72%)',
          }}
        />
        <div className="relative mx-auto flex max-w-4xl flex-col items-center px-6 py-20 text-center">
          <p
            className="mb-6 font-mono text-[0.7rem] uppercase tracking-[0.35em]"
            style={{ color: 'hsl(176 55% 62%)' }}
          >
            Population PK · Model Discovery
          </p>

          <Image
            src="/apmode-logo.png"
            alt="APMODE — Adaptive Pharmacokinetic Model Discovery Engine"
            width={1536}
            height={1024}
            priority
            className="h-auto w-full max-w-[440px] drop-shadow-2xl"
          />

          <p className="mt-6 max-w-2xl text-lg text-balance text-white/80">
            A governed meta-system that composes five population-PK modeling
            paradigms into a single, auditable model-discovery workflow.
          </p>

          <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
            <Link
              href="/docs"
              className="rounded-lg bg-fd-primary px-5 py-2.5 text-sm font-semibold text-fd-primary-foreground transition-transform hover:-translate-y-0.5"
            >
              Read the docs
            </Link>
            <Link
              href="/docs/guide/getting-started/quickstart"
              className="rounded-lg border border-white/20 px-5 py-2.5 text-sm font-semibold text-white/90 transition-colors hover:bg-white/5"
            >
              Quickstart →
            </Link>
          </div>

          <p className="mt-8 font-mono text-xs tracking-wide text-white/40">
            v0.6.1-rc2 · GPL-2.0-or-later · nlmixr2 · Stan/Torsten · JAX/Diffrax
          </p>
        </div>
      </section>

      {/* ---- Five paradigms strip ---- */}
      <section className="border-b border-fd-border bg-fd-muted/40">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-center gap-x-6 gap-y-2 px-6 py-5">
          <span className="font-mono text-xs uppercase tracking-widest text-fd-muted-foreground">
            Five paradigms, one pipeline
          </span>
          {PARADIGMS.map((p) => (
            <span
              key={p}
              className="flex items-center gap-2 text-sm text-fd-foreground/80"
            >
              <span
                aria-hidden
                className="inline-block h-1.5 w-1.5 rounded-full bg-fd-primary"
              />
              {p}
            </span>
          ))}
        </div>
      </section>

      {/* ---- Feature grid ---- */}
      <section className="mx-auto w-full max-w-5xl px-6 py-16">
        <h2 className="mb-2 text-2xl font-semibold tracking-tight">
          What makes it different
        </h2>
        <p className="mb-8 max-w-2xl text-fd-muted-foreground">
          APMODE is not a wrapper around one fitter — it is a governed funnel that
          holds every paradigm to the same evidence standard.
        </p>
        <div className="grid gap-4 sm:grid-cols-2">
          {FEATURES.map((f) => (
            <Link
              key={f.title}
              href={f.href}
              className="group relative flex flex-col rounded-xl border border-fd-border bg-fd-card p-6 transition-colors hover:border-fd-primary/50"
            >
              <span
                aria-hidden
                className="absolute left-0 top-6 h-8 w-[3px] rounded-r bg-fd-primary opacity-70"
              />
              <h3 className="mb-2 font-semibold tracking-tight">{f.title}</h3>
              <p className="mb-4 flex-1 text-sm leading-relaxed text-fd-muted-foreground">
                {f.body}
              </p>
              <span className="text-sm font-medium text-fd-primary">
                {f.cta}{' '}
                <span className="inline-block transition-transform group-hover:translate-x-1">
                  →
                </span>
              </span>
            </Link>
          ))}
        </div>
      </section>
    </main>
  );
}
