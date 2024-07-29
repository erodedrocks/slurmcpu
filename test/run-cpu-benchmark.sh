#!/usr/bin/env sh

#SBATCH --job-name=run-cpu-benchmark
#SBATCH --account=ddp324
#SBATCH --clusters=expanse
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x.o%A.%a.%N

export MKL_SERVICE_FORCE_INTEL=1

declare -xir UNIX_TIME="$(date +'%s')"
declare -xr LOCAL_TIME="$(date +'%Y%m%dT%H%M%S%z')"

declare -xr SLURM_JOB_SCRIPT="$(scontrol show job ${SLURM_JOB_ID} | awk -F= '/Command=/{print $2}')"
declare -xr SLURM_JOB_SCRIPT_MD5="$(md5sum ${SLURM_JOB_SCRIPT} | awk '{print $1}')"
declare -xr SLURM_JOB_SCRIPT_SHA256="$(sha256sum ${SLURM_JOB_SCRIPT} | awk '{print $1}')"
declare -xr SLURM_JOB_SCRIPT_NUMBER_OF_LINES="$(wc -l ${SLURM_JOB_SCRIPT} | awk '{print $1}')"

declare -xr LUSTRE_PROJECT_DIR="/expanse/lustre/projects/${SLURM_JOB_ACCOUNT}/${USER}"
declare -xr LUSTRE_SCRATCH_DIR="/expanse/lustre/scratch/${USER}/temp_project"
declare -xr LOCAL_SCRATCH_DIR="/scratch/${USER}/job_${SLURM_JOB_ID}"
declare -xr CEPH_USER_DIR="/expanse/ceph/users/${USER}"

declare -xr CONDA_CACHE_DIR="${SLURM_SUBMIT_DIR}"
declare -xr CONDA_ENV_YAML="${CONDA_CACHE_DIR}/environments/benchmark.yaml"
declare -xr CONDA_ENV_NAME="$(grep '^name:' ${CONDA_ENV_YAML} | awk '{print $2}')"

echo "${UNIX_TIME} ${LOCAL_TIME} ${SLURM_JOB_ID} ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} ${SLURM_JOB_SCRIPT_MD5} ${SLURM_JOB_SCRIPT_SHA256} ${SLURM_JOB_SCRIPT_NUMBER_OF_LINES}"
cat "${SLURM_JOB_SCRIPT}"

module purge
module load slurm
module list

cd "${LOCAL_SCRATCH_DIR}"

md5sum -c "${CONDA_ENV_YAML}.md5"
if [[ "${?}" -eq 0 ]]; then

  echo "Unpacking existing the conda environment to ${LOCAL_SCRATCH_DIR} ..."
  cp "${CONDA_CACHE_DIR}/${CONDA_ENV_NAME}.tar.gz" ./
  tar -xf "${CONDA_ENV_NAME}.tar.gz"
  source bin/activate
  conda-unpack

else

  echo "Installing miniconda to ${LOCAL_SCRATCH_DIR} ..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  chmod +x Miniconda3-latest-Linux-x86_64.sh
  export CONDA_INSTALL_PATH="${LOCAL_SCRATCH_DIR}/miniconda3"
  export CONDA_ENVS_PATH="${CONDA_INSTALL_PATH}/envs"
  export CONDA_PKGS_DIRS="${CONDA_INSTALL_PATH}/pkgs"
  ./Miniconda3-latest-Linux-x86_64.sh -b -p "${CONDA_INSTALL_PATH}"

  echo "Re/building the conda environment from ${CONDA_ENV_YAML} ..."
  source "${CONDA_INSTALL_PATH}/etc/profile.d/conda.sh"
  conda activate base
  conda install -y mamba -n base -c conda-forge
  mamba env create --file "${CONDA_ENV_YAML}"
  conda install -y conda-pack

  echo "Packing the conda environment and caching it to ${CONDA_CACHE_DIR} ..."
  conda pack -n "${CONDA_ENV_NAME}" -o "${CONDA_ENV_NAME}.tar.gz"
  cp "${CONDA_ENV_NAME}.tar.gz" "${CONDA_CACHE_DIR}"
  md5sum "${CONDA_ENV_YAML}" > "${CONDA_ENV_YAML}.md5"
  conda activate "${CONDA_ENV_NAME}"

fi

echo "Finalizing the software environment configuration ..."
printenv

cd "${SLURM_SUBMIT_DIR}"
cd ".."

python3 cpu-benchmark.py -n "torch-model-training" -c "1,8,16,32" -j 16 -o 6 -i 4.5 -f "goal_speedup" -b 0.05

echo "Job completed"