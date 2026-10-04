#!/usr/bin/env bash
# Reproduces everything in results/ppo/. Two lanes run side by side (one CPU thread each).
# Rough cost per run on one laptop-class core: chaotic ring 25 min (CartPole) / 70 min (Pendulum);
# stationary rings 10 min / 35 min; ring-free policies: seconds.
#
#   bash run_experiments.sh cartpole    # rings in all four regimes vs MLP vs ring removed, 3 seeds
#   bash run_experiments.sh ablation    # chaotic ring on CartPole: encoding, T_avg, eps, readout width, noise control
#   bash run_experiments.sh pendulum    # swing-up: linear fails, quadratic features and the rings do not
#   bash run_experiments.sh lunar       # LunarLander: 8 inputs, 4 actions; linear fails, quadratic / MLP / ring solve it
#   bash run_experiments.sh topo        # coupled-ring lattice (topological frequency comb) on CartPole, 3 seeds (~1 h/run)
#   bash run_experiments.sh topo2       # the lattice on the two harder tasks: Pendulum swing-up + LunarLander, 3 seeds each
#   bash run_experiments.sh topo_chaos  # the lattice past MI threshold: a self-generated chaotic topological comb on CartPole
#   bash run_experiments.sh topo_chaos2 # the chaotic comb on Pendulum + LunarLander, 3 seeds each (long: T_avg = 25)
# Every run is resumable: an interrupted run continues from its last checkpoint when the script is re-run.
set -u
PY=${PYTHON:-python}                       # e.g. PYTHON=python3 bash run_experiments.sh cartpole
# On macOS, hold off idle sleep for as long as this script (and its lanes) is alive.
command -v caffeinate > /dev/null && { caffeinate -is -w $$ & }
run() {  # run <Env> <result-name> <args...>   (skips runs whose result file already exists)
    local env=$1 name=$2; shift 2
    mkdir -p "results/ppo/${env}/logs"
    [ -f "results/ppo/${env}/${name}.json" ] && { echo "skip ${env}/${name}"; return; }
    local ver=v1; [ "$env" = "LunarLander" ] && ver=v3
    $PY PPO_MR.py --env "${env}-${ver}" --resume "$@" >> "results/ppo/${env}/logs/${name}.log" 2>&1
    echo "done ${env}/${name}: $(grep greedy results/ppo/${env}/logs/${name}.log)"
}

case "${1:-cartpole}" in
cartpole)
    ( for s in 0 1 2; do run CartPole mr_chaos_seed$s --policy mr --regime chaos --seed $s; done
      run CartPole mr_rolls_seed0 --policy mr --regime rolls --seed 0 ) &
    ( for s in 0 1 2; do run CartPole mr_normal_seed$s --policy mr --regime normal --seed $s; done
      run CartPole mr_soliton_eps0.02_seed0 --policy mr --regime soliton --seed 0 --tag eps0.02
      run CartPole mr_soliton_eps0.02_nodet_seed0 --policy mr --regime soliton --seed 0 --tag eps0.02_nodet --detector_noise 0
      for p in nn linear; do for s in 0 1 2; do run CartPole ${p}_seed$s --policy $p --seed $s; done; done ) &
    wait; $PY compare_policies.py --env CartPole; $PY compare_policies.py --readout ;;
ablation)
    ( run CartPole mr_chaos_signed_seed0  --policy mr --seed 0 --encoding signed --tag signed
      run CartPole mr_chaos_eps0.3_seed0  --policy mr --seed 0 --eps 0.3 --tag eps0.3
      run CartPole mr_chaos_eps1.0_seed0  --policy mr --seed 0 --eps 1.0 --tag eps1.0
      run CartPole mr_chaos_Tavg100_seed0 --policy mr --seed 0 --T_avg 100 --tag Tavg100 --n_updates 40 ) &   # 4x the ring time per step
    ( for s in 0 1 2; do run CartPole mr_chaos_Tavg5_seed$s  --policy mr --seed $s --T_avg 5  --tag Tavg5
                         run CartPole mr_chaos_Tavg10_seed$s --policy mr --seed $s --T_avg 10 --tag Tavg10; done
      run CartPole mr_chaos_allmodes_seed0 --policy mr --seed 0 --readout_halfwidth 0 --tag allmodes
      for n in 0.05 0.1 0.2 0.4; do for s in 0 1 2; do run CartPole linear_noise${n}_seed$s --policy linear --obs_noise $n --tag noise$n --seed $s; done; done ) &
    wait; $PY compare_policies.py --ablation ;;
pendulum)
    # --readout_halfwidth 8: the published Pendulum rings read the 17 lines |m| <= 8 (54 weights),
    # not the 2-per-input default that 3 inputs would give
    ( for s in 0 1 2; do run Pendulum mr_normal_seed$s --policy mr --regime normal --seed $s --readout_halfwidth 8; done ) &
    ( run Pendulum mr_chaos_seed0 --policy mr --regime chaos --seed 0 --readout_halfwidth 8 --eps 1.0 --n_updates 200
      for p in nn linear poly2; do for s in 0 1 2; do run Pendulum ${p}_seed$s --policy $p --seed $s; done; done ) &
    wait; $PY compare_policies.py --env Pendulum ;;
topo)
    ( run CartPole mr_topo_seed0 --policy mr --regime topo --seed 0
      run CartPole mr_topo_seed2 --policy mr --regime topo --seed 2 ) &
    ( run CartPole mr_topo_seed1 --policy mr --regime topo --seed 1 ) &
    wait; $PY compare_policies.py --env CartPole ;;
topo_chaos)
    ( run CartPole mr_topo_chaos_seed0 --policy mr --regime topo_chaos --seed 0
      run CartPole mr_topo_chaos_seed2 --policy mr --regime topo_chaos --seed 2 ) &
    ( run CartPole mr_topo_chaos_seed1 --policy mr --regime topo_chaos --seed 1 ) &
    wait; $PY compare_policies.py --env CartPole ;;
topo_chaos2)
    ( run Pendulum mr_topo_chaos_seed0 --policy mr --regime topo_chaos --seed 0 --readout_halfwidth 8
      run Pendulum mr_topo_chaos_seed2 --policy mr --regime topo_chaos --seed 2 --readout_halfwidth 8 ) &
    ( run Pendulum mr_topo_chaos_seed1 --policy mr --regime topo_chaos --seed 1 --readout_halfwidth 8 ) &
    ( run LunarLander mr_topo_chaos_seed0 --policy mr --regime topo_chaos --seed 0 --n_updates 250
      run LunarLander mr_topo_chaos_seed2 --policy mr --regime topo_chaos --seed 2 --n_updates 250 ) &
    ( run LunarLander mr_topo_chaos_seed1 --policy mr --regime topo_chaos --seed 1 --n_updates 250 ) &
    wait; $PY compare_policies.py --env Pendulum; $PY compare_policies.py --env LunarLander ;;
topo2)
    # Pendulum rings read |m| <= 8 like the published mr_normal runs (see the pendulum case)
    ( for s in 0 1 2; do run Pendulum mr_topo_seed$s --policy mr --regime topo --seed $s --readout_halfwidth 8; done
      $PY compare_policies.py --env Pendulum ) &
    ( for s in 0 1 2; do run LunarLander mr_topo_seed$s --policy mr --regime topo --seed $s --n_updates 250; done
      $PY compare_policies.py --env LunarLander ) &
    wait ;;
lunar)
    ( for s in 0 1 2; do run LunarLander mr_normal_seed$s --policy mr --regime normal --seed $s --n_updates 250; done ) &
    ( run LunarLander mr_chaos_seed0 --policy mr --regime chaos --seed 0 --eps 1.0 --n_updates 250
      for p in nn linear poly2; do for s in 0 1 2; do run LunarLander ${p}_seed$s --policy $p --seed $s; done; done ) &
    wait; $PY compare_policies.py --env LunarLander ;;
esac
