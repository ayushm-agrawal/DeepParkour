#!/bin/sh
#SBATCH --time=128:00:00          # Run time in hh:mm:ss
#SBATCH --mem=50000
#SBATCH --job-name=run_agent_4
#SBATCH --partition=cse496dl
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --output=/work/cse496dl/teams/Dropouts/final_project/out_files/50M_agent.out

python -u $@ --timesteps=10000 --env=HumanoidBulletEnv-v0 --model-path=/work/cse496dl/teams/Dropouts/final_project/agents/50M_agent/humanoid_policy
