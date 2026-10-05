#!/bin/bash

set -euo pipefail
umask 022

RUNNER="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment/run_vindr_symmetric_r20_top20_calibrated.sh"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_symmetric_r20_top20_calibrated/20260827_v1"
JOB_NAME="v-r20-cal-top20"
EXPECTED_USER="yz3522"

PAUSE_JOB="279395"
PAUSE_JOB_NAME="mb-top20-group"
PAUSE_JOB_COMMAND="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment/run_own_top20_unified_downstream_group.sh"
PAUSE_SAFE_MARKER="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_own_top20_binary_v5_llm_refine_5seed/20260714_065539/seed_7/llm_refine/loop_05/.downstream_mobilenet100_es10_best_complete"

PENDING_COMPETITORS=(279396 279486)
declare -A EXPECTED_COMPETITOR_COMMAND=(
  [279396]="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment/run_own_top20_unified_downstream_group.sh"
  [279486]="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment/run_mobilenetv3_unified_downstream_seed.sh"
)

for command_name in sbatch scancel scontrol squeue; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "Required Slurm command is unavailable: ${command_name}" >&2
    exit 2
  }
done

CURRENT_USER="$(id -un)"
[[ "${CURRENT_USER}" == "${EXPECTED_USER}" ]] || {
  echo "Expected user ${EXPECTED_USER}; found ${CURRENT_USER}" >&2
  exit 2
}
[[ -x "${RUNNER}" ]] || {
  echo "Runner is missing or not executable: ${RUNNER}" >&2
  exit 2
}
[[ -e "${OUTPUT_ROOT}/prepared/.prepare_complete" ]] || {
  echo "Prepared-data completion marker is missing" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}/.job_complete" ]] || {
  echo "The calibrated top-20% experiment is already complete" >&2
  exit 2
}
[[ -e "${PAUSE_SAFE_MARKER}" ]] || {
  echo "The durable checkpoint preceding the resumable job is missing" >&2
  exit 2
}

job_field() {
  local details="$1"
  local field="$2"
  printf '%s\n' "${details}" | tr ' ' '\n' | sed -n "s/^${field}=//p" | head -n 1
}

append_gate() {
  local dependency="$1"
  local new_job="$2"
  if [[ -z "${dependency}" || "${dependency}" == "(null)" ]]; then
    printf 'afterany:%s\n' "${new_job}"
  else
    printf '%s,afterany:%s\n' "${dependency}" "${new_job}"
  fi
}

restore_dependency() {
  local job_id="$1"
  local dependency="$2"
  if [[ -z "${dependency}" || "${dependency}" == "(null)" ]]; then
    scontrol update JobId="${job_id}" Dependency=
  else
    scontrol update JobId="${job_id}" Dependency="${dependency}"
  fi
}

EXISTING_NEW_JOB="$(squeue -h -u "${CURRENT_USER}" -o '%i|%j|%T' | awk -F'|' -v name="${JOB_NAME}" '$2 == name {print}')"
[[ -z "${EXISTING_NEW_JOB}" ]] || {
  echo "A live ${JOB_NAME} job already exists:" >&2
  echo "${EXISTING_NEW_JOB}" >&2
  exit 2
}

PAUSE_DETAILS="$(scontrol show job -o "${PAUSE_JOB}" 2>/dev/null || true)"
[[ -n "${PAUSE_DETAILS}" ]] || {
  echo "Pause target ${PAUSE_JOB} is no longer live; no job was changed" >&2
  exit 2
}
[[ "$(job_field "${PAUSE_DETAILS}" JobName)" == "${PAUSE_JOB_NAME}" ]] || {
  echo "Pause target job name is not the locked expected job" >&2
  exit 2
}
[[ "$(job_field "${PAUSE_DETAILS}" Command)" == "${PAUSE_JOB_COMMAND}" ]] || {
  echo "Pause target command is not the locked resumable runner" >&2
  exit 2
}
[[ "$(job_field "${PAUSE_DETAILS}" JobState)" == "RUNNING" ]] || {
  echo "Pause target ${PAUSE_JOB} is not currently running; no job was changed" >&2
  exit 2
}
[[ "$(job_field "${PAUSE_DETAILS}" Partition)" == "a16" ]] || {
  echo "Pause target is not using the expected A16 partition" >&2
  exit 2
}
[[ "$(job_field "${PAUSE_DETAILS}" UserId)" == "${CURRENT_USER}"\(* ]] || {
  echo "Pause target is not owned by ${CURRENT_USER}" >&2
  exit 2
}
[[ "$(job_field "${PAUSE_DETAILS}" Requeue)" == "1" ]] || {
  echo "Pause target is not marked as requeueable" >&2
  exit 2
}
PAUSE_ORIGINAL_DEPENDENCY="$(job_field "${PAUSE_DETAILS}" Dependency)"
[[ "${PAUSE_ORIGINAL_DEPENDENCY}" != *'?'* ]] || {
  echo "Pause target has an OR dependency that cannot be safely extended" >&2
  exit 2
}

declare -a LIVE_PENDING_COMPETITORS=()
declare -A ORIGINAL_DEPENDENCY=()
for pending_job in "${PENDING_COMPETITORS[@]}"; do
  details="$(scontrol show job -o "${pending_job}" 2>/dev/null || true)"
  [[ -n "${details}" ]] || continue
  [[ "$(job_field "${details}" JobState)" == "PENDING" ]] || continue
  [[ "$(job_field "${details}" Partition)" == "a16" ]] || {
    echo "Pending competitor ${pending_job} is not on A16" >&2
    exit 2
  }
  [[ "$(job_field "${details}" UserId)" == "${CURRENT_USER}"\(* ]] || {
    echo "Pending competitor ${pending_job} is not owned by ${CURRENT_USER}" >&2
    exit 2
  }
  [[ "$(job_field "${details}" Command)" == "${EXPECTED_COMPETITOR_COMMAND[${pending_job}]}" ]] || {
    echo "Pending competitor ${pending_job} has an unexpected command" >&2
    exit 2
  }
  dependency="$(job_field "${details}" Dependency)"
  [[ "${dependency}" != *'?'* ]] || {
    echo "Pending competitor ${pending_job} has an OR dependency that cannot be safely extended" >&2
    exit 2
  }
  LIVE_PENDING_COMPETITORS+=("${pending_job}")
  ORIGINAL_DEPENDENCY[${pending_job}]="${dependency}"
done
ORIGINAL_DEPENDENCY[${PAUSE_JOB}]="${PAUSE_ORIGINAL_DEPENDENCY}"

echo "Submission preflight"
echo "Runner: ${RUNNER}"
echo "Verified resumable A16 pause target: ${PAUSE_JOB} (${PAUSE_JOB_NAME})"
squeue -u "${CURRENT_USER}" -o '%.18i %.9P %.30j %.2t %.10M %.10l %R'
sbatch --test-only "${RUNNER}"

NEW_JOB=""
declare -a TO_RESTORE=()
rollback() {
  local status="$?"
  trap - ERR
  set +e
  echo "Submission transaction failed; restoring displaced jobs" >&2
  for job_id in "${TO_RESTORE[@]}"; do
    restore_dependency "${job_id}" "${ORIGINAL_DEPENDENCY[${job_id}]}" >/dev/null 2>&1 || true
    scontrol release "${job_id}" >/dev/null 2>&1 || true
  done
  if [[ "${NEW_JOB}" =~ ^[0-9]+$ ]]; then
    scancel "${NEW_JOB}" >/dev/null 2>&1 || true
  fi
  echo "Rollback completed; inspect the queue before retrying" >&2
  exit "${status}"
}
trap rollback ERR

for pending_job in "${LIVE_PENDING_COMPETITORS[@]}"; do
  scontrol hold "${pending_job}"
  TO_RESTORE+=("${pending_job}")
done

SUBMITTED="$(sbatch --parsable "${RUNNER}")"
NEW_JOB="${SUBMITTED%%;*}"
[[ "${NEW_JOB}" =~ ^[0-9]+$ ]]
echo "Submitted calibrated top-20% experiment as job ${NEW_JOB}"

for pending_job in "${LIVE_PENDING_COMPETITORS[@]}"; do
  combined_dependency="$(append_gate "${ORIGINAL_DEPENDENCY[${pending_job}]}" "${NEW_JOB}")"
  scontrol update JobId="${pending_job}" Dependency="${combined_dependency}"
  scontrol release "${pending_job}"
done

scontrol requeuehold "${PAUSE_JOB}"
TO_RESTORE+=("${PAUSE_JOB}")
PAUSE_DEPENDENCY="$(append_gate "${PAUSE_ORIGINAL_DEPENDENCY}" "${NEW_JOB}")"
scontrol update JobId="${PAUSE_JOB}" Dependency="${PAUSE_DEPENDENCY}"
scontrol release "${PAUSE_JOB}"

trap - ERR
echo "Paused job ${PAUSE_JOB}; it will resume automatically after job ${NEW_JOB} ends"
SUBMISSION_RECORD="${OUTPUT_ROOT}/submission_job_${NEW_JOB}.txt"
printf 'submitted_at=%s\nnew_job_id=%s\npaused_job_id=%s\ngated_pending_jobs=%s\nrunner=%s\n' \
  "$(date --iso-8601=seconds)" "${NEW_JOB}" "${PAUSE_JOB}" \
  "${LIVE_PENDING_COMPETITORS[*]}" "${RUNNER}" > "${SUBMISSION_RECORD}"
squeue -u "${CURRENT_USER}" -o '%.18i %.9P %.30j %.2t %.10M %.10l %R'
printf 'NEW_JOB_ID=%s\nPAUSED_JOB_ID=%s\n' "${NEW_JOB}" "${PAUSE_JOB}"
