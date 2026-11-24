"""Unified command-line interface for Ising simulations and experiments."""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np

from .model import IsingModel, validate_algorithm, DYNAMICS
from .experiments import run, field_scan, equilibrium_check, first_passage, coalescence as coupling_trials
from .visualization.animation import save_animation
from .benchmarks.benchmark import run_benchmark


# Argument validators

def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return n


def nonnegative(value):
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return n


# Command handlers

def cmd_series(a):
    _, series = run(IsingModel(a.N,a.beta,a.lam), a.sweeps, init=a.init,seed=a.seed,dynamics=a.dynamics,backend=a.backend,record_every=a.record_every or max(1,a.sweeps//40))
    for row in series:
        m = row['magnetization']
        print(f"{row['sweep']:7g}  m={m:+.3f}  " + '#' * int((m+1)*30))


def cmd_scan(a):
    print('lambda dynamics init mean_m_second_half largest_plus_fraction')
    for row in field_scan(a.N,a.beta,a.lams,a.sweeps,seed=a.seed,backend=a.backend):
        print(f"{row['lam']:g} {row['dynamics']} {row['init']} {row['mean_m']:+.3f} {row['largest_plus_fraction']:.3f}")


def cmd_check(a):
    exact, rows = equilibrium_check(IsingModel(a.N,a.beta,a.lam),a.steps,a.burn,seed=a.seed,backend=a.backend)
    print('Exact:', exact)
    for row in rows:
        print(f"{row['dynamics']:>10} {row['observable']:>10}: {row['mean']:.5f} +/- {row['se']:.5f}; z={row['z']}")


def cmd_movie(a):
    if a.out.exists():
        raise FileExistsError(a.out)
    save_animation(IsingModel(a.N,a.beta,a.lam),a.out,dynamics=a.dynamics,backend=a.backend,init=a.init,seed=a.seed,frames=a.frames,stride=a.stride)
    print(f'Wrote {a.out}')


def progress(done,total,row):
    print(f"Trial {done}/{total}: N={row['N']}, t={row['sweeps']:g}, event={row['event']}",flush=True)


def cmd_passage(a):
    first_passage(a.N,beta=a.beta,lam=a.lam,dynamics=a.dynamics,backend=a.backend,init=a.init,threshold=a.threshold,limit=a.limit,replicates=a.replicates,seed=a.seed,workers=a.workers,output=a.out,resume=a.resume,progress=progress)
    print(f'Wrote {a.out}/trials.csv')


def cmd_coalescence(a):
    coupling_trials(a.N,beta=a.beta,lam=a.lam,backend=a.backend,limit=a.limit,replicates=a.replicates,seed=a.seed,workers=a.workers,output=a.out,resume=a.resume,progress=progress)
    print(f'Wrote {a.out}/trials.csv')


def cmd_benchmark(a):
    rows = run_benchmark(a.N)
    writer = csv.DictWriter(sys.stdout,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


# Parser construction

def build_parser():
    parser = argparse.ArgumentParser(
        prog='ising',
        description='2D Ising model with fixed minus boundary and field h = lambda/N.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  ising series --size 64 --beta .5 --field 3
  ising movie --size 64 --output results/bubble.gif
  ising passage --size 32 48 64 --replicates 20 --workers 4
  ising coalescence --size 32 --replicates 10

Run ising COMMAND --help for options and defaults.''')
    sub = parser.add_subparsers(dest='cmd', metavar='COMMAND')

    def command(name, description, handler):
        p = sub.add_parser(name, help=description, description=description,
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.set_defaults(handler=handler)
        return p

    def model(p, size=48, beta=.6, field=2., multiple=False, scan=False,
              dynamics=False, init=None, seed=0):
        g = p.add_argument_group('Model')
        kw = {'nargs': '+'} if multiple else {}
        g.add_argument('-N', '--size', '--N', dest='N', type=positive, default=size,
                       help='interior lattice side length', **kw)
        temp = g.add_mutually_exclusive_group()
        temp.add_argument('--beta', type=float, default=beta, help='inverse temperature')
        temp.add_argument('--temperature', type=float, help='temperature T = 1/beta')
        if scan:
            g.add_argument('--fields', '--lams', dest='lams', type=float, nargs='+',
                           default=[.5, 1, 1.5, 2, 2.5, 3], help='rescaled fields lambda')
        else:
            g.add_argument('--field', '--lambda', '--lam', dest='lam', type=float,
                           default=field, help='rescaled field lambda (physical field = lambda/N)')
        r = p.add_argument_group('Dynamics and reproducibility')
        if dynamics:
            r.add_argument('--dynamics', choices=DYNAMICS,
                           default='heatbath', help='transition kernel; sw-ghost and sw-size are cluster variants')
        if init:
            r.add_argument('--init', choices=init, default='minus', help='initial spin configuration')
        r.add_argument('--backend', choices=['numba', 'python'], default='numba',
                       help='compiled or reference kernel')
        r.add_argument('--seed', type=nonnegative, default=seed, help='random seed')

    p = command('series', 'Print a magnetization trajectory.', cmd_series)
    model(p, dynamics=True, init=['minus', 'plus', 'random'])
    p.add_argument('--sweeps', type=positive, default=4000, help='duration in sweeps')
    p.add_argument('--record-every', type=positive, help='sampling interval; default approximately 40 observations')
    p = command('scan', 'Compare fields, dynamics and extremal starts.', cmd_scan)
    model(p, scan=True)
    p.add_argument('--sweeps', type=positive, default=4000, help='sweeps per trajectory')
    p = command('movie', 'Save a spin-lattice animation as a GIF.', cmd_movie)
    model(p, dynamics=True, init=['minus', 'plus', 'random'])
    p.add_argument('--frames', type=positive, default=150, help='number of animation frames')
    p.add_argument('--stride', type=positive, default=5, help='sweeps between frames')
    p.add_argument('-o', '--output', '--out', dest='out', type=Path, default=Path('bubble.gif'), help='output GIF path')
    p = command('check', 'Compare Monte Carlo estimates with exact Gibbs probabilities.', cmd_check)
    model(p, size=3, seed=1)
    p.add_argument('--steps', type=positive, default=100000, help='observed sweeps per dynamics (at least 20)')
    p.add_argument('--burn', type=nonnegative, default=500, help='discarded initial sweeps')
    for name, handler, description, sizes, reps, seed, output in [
        ('passage', cmd_passage, 'Measure replicated magnetization first-passage times.', [32,48,64], 10, 12345, 'results/passage'),
        ('coalescence', cmd_coalescence, 'Measure extremal heat-bath coupling times.', [32], 20, 0, 'results/coalescence')]:
        p = command(name, description, handler)
        model(p, size=sizes, beta=.5, field=3., multiple=True, seed=seed,
              dynamics=name=='passage', init=['minus','plus'] if name=='passage' else None)
        p.add_argument('--limit', type=positive if name=='passage' else nonnegative, default=10000, help='maximum sweeps per replicate')
        p.add_argument('--replicates', type=positive, default=reps, help='independent trials per size')
        p.add_argument('-o','--output','--out',dest='out',type=Path,default=Path(output),help='new study directory (or existing with --resume)')
        if name=='passage':
            p.add_argument('--threshold',type=float,default=0.,help='magnetization threshold; upward from minus, downward from plus')
        p.add_argument('--workers',type=positive,default=1,help='parallel worker processes')
        p.add_argument('--resume',action='store_true',help='resume an existing matching study')
    p = command('benchmark', 'Compare Python and Numba sweep timings for all dynamics.', cmd_benchmark)
    p.add_argument('-N','--size','--N',dest='N',type=positive,nargs='+',default=[32,64,128,256],help='lattice sizes')
    return parser


# Entry point

def main(argv=None):
    parser = build_parser()
    a = parser.parse_args(argv)
    if a.cmd is None:
        parser.print_help()
        return
    if hasattr(a, 'beta'):
        try:
            if a.temperature is not None:
                if not np.isfinite(a.temperature) or a.temperature <= 0:
                    raise ValueError('temperature must be finite and positive')
                a.beta = 1 / a.temperature
            for n in a.N if isinstance(a.N, list) else [a.N]:
                for lam in getattr(a, 'lams', [getattr(a, 'lam', 0)]):
                    IsingModel(n, a.beta, lam)
                    validate_algorithm(getattr(a, 'dynamics', 'heatbath'), a.backend)
            if a.cmd == 'check' and (a.N > 4 or a.steps < 20):
                raise ValueError('check requires size <= 4 and steps >= 20')
            if a.cmd == 'passage' and not -1 < a.threshold < 1:
                raise ValueError('threshold must be strictly between -1 and 1')
            if a.cmd == 'movie' and a.out.suffix.lower() != '.gif':
                raise ValueError('movie output must end in .gif')
            if hasattr(a, 'out'):
                a.out.parent.mkdir(parents=True, exist_ok=True)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    try:
        a.handler(a)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
