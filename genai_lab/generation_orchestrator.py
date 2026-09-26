"""GUI, CLI와 GPU 검증이 공유하는 제품 이미지 생성 애플리케이션 서비스."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
import json
import shutil
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable


class GenerationPhase(str, Enum):
    CREATED = "created"
    INPUTS_PREPARED = "inputs_prepared"
    INPUTS_APPROVED = "inputs_approved"
    BASE_RUNNING = "base_running"
    BASE_COMPLETED = "base_completed"
    CANDIDATE_SELECTED = "candidate_selected"
    FINAL_RUNNING = "final_running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerationOrchestrationError(RuntimeError):
    """제품 생성 단계를 잘못된 순서로 호출했을 때 발생한다."""


@dataclass(frozen=True)
class GenerationExecutionEvidence:
    path: Path
    record: dict[str, Any]


@dataclass(frozen=True)
class BaseCandidateSelection:
    """오케스트레이터가 검증한 Native 인계용 Base 후보 파일 묶음."""

    index: int
    candidate_number: int
    base_image: Path
    candidate_record: Path
    approved_run: Path
    garment_reference: Path
    batch_directory: Path
    input_mode: str = "approved_base"


class GenerationOrchestrator:
    """입력 준비, 승인 계약, Base 생성과 Native 실행 순서를 한 곳에서 관리한다."""

    VERSION = "product_generation_orchestrator_v3"
    MANIFEST_NAME = "product_execution.json"

    def __init__(
        self,
        config: dict[str, Any],
        request: Any,
        project_root: Path,
        run_log: Any,
        *,
        cancelled: Callable[[], bool] = lambda: False,
        status_callback: Callable[[str], None] = lambda _message: None,
        prepare_visual_inputs_fn: Callable[..., Any] | None = None,
        generate_visual_batch_fn: Callable[..., Any] | None = None,
        approve_reference_run_fn: Callable[..., Any] | None = None,
        require_reference_run_fn: Callable[..., Any] | None = None,
        save_replay_bundle_fn: Callable[..., Path] | None = None,
    ) -> None:
        if prepare_visual_inputs_fn is None or generate_visual_batch_fn is None:
            from genai_lab.visual_reference import (
                generate_visual_batch,
                prepare_visual_inputs,
            )
            prepare_visual_inputs_fn = (
                prepare_visual_inputs_fn or prepare_visual_inputs
            )
            generate_visual_batch_fn = (
                generate_visual_batch_fn or generate_visual_batch
            )
        if approve_reference_run_fn is None or require_reference_run_fn is None:
            from genai_lab.approved_reference_run import (
                approve_reference_run,
                require_reference_run,
            )
            approve_reference_run_fn = (
                approve_reference_run_fn or approve_reference_run
            )
            require_reference_run_fn = (
                require_reference_run_fn or require_reference_run
            )
        from genai_lab.provenance import ensure_recorder
        ensure_recorder(config, root=project_root)
        self.config = config
        self.request = request
        self.project_root = Path(project_root)
        self.run_log = run_log
        self.cancelled = cancelled
        self.status_callback = status_callback
        self._prepare_visual_inputs = prepare_visual_inputs_fn
        self._generate_visual_batch = generate_visual_batch_fn
        self._approve_reference_run = approve_reference_run_fn
        if save_replay_bundle_fn is None:
            from genai_lab.generation_replay import (
                save_generation_replay_bundle,
            )
            save_replay_bundle_fn = save_generation_replay_bundle
        self._require_reference_run = require_reference_run_fn
        self._save_replay_bundle = save_replay_bundle_fn
        self.phase = GenerationPhase.CREATED
        self.trace: list[dict[str, Any]] = []
        self._prepared_inputs: Any | None = None
        self._base_batch: Any | None = None
        self._selected_base: BaseCandidateSelection | None = None
        self._generation_pipeline: Any | None = None
        self._final_review_evidence: Any | None = None
        self._final_review_output_directory: Path | None = None
        from genai_lab.native_pipeline_contract import resolve_refinement_mode
        from genai_lab.generation_run_context import GenerationRunContext
        self.refinement_mode = resolve_refinement_mode(self.config)
        self.run_context = GenerationRunContext.create(
            self.config,
            self.project_root,
            refinement_mode=self.refinement_mode,
        )
        from genai_lab.provenance import recorder
        recorder(self.config).bind(self.run_context)
        self.replay_bundle: Path | None = None
        self._record(
            "run_context",
            "created",
            run_id=self.run_context.run_id,
            run_directory=str(self.run_context.run_directory),
            refinement_mode=self.refinement_mode,
        )

    def _require_phase(self, expected: GenerationPhase) -> None:
        if self.phase is not expected:
            raise GenerationOrchestrationError(
                f"생성 단계 순서 오류: 현재={self.phase.value}, "
                f"필요={expected.value}"
            )

    def _record(self, stage: str, status: str, **details: Any) -> None:
        event = {"stage": stage, "status": status}
        event.update(details)
        self.trace.append(event)
        from genai_lab.provenance import recorder
        recorder(self.config).event(stage, status, **details)
        if self.run_log is not None:
            self.run_log.write_stage(
                "공통 생성 오케스트레이터",
                json.dumps(event, ensure_ascii=False, sort_keys=True),
            )

    def close(self) -> None:
        """Close the request-owned log after all GUI review stages finish."""
        if self.run_log is None:
            return
        close_log = getattr(self.run_log, "close", None)
        if callable(close_log):
            close_log()
        self.run_log = None

    def prepare_visual_inputs(self, source: Any, garment: Any) -> Any:
        self._require_phase(GenerationPhase.CREATED)
        self._record("prepare_visual_inputs", "started")
        try:
            prepared = self._prepare_visual_inputs(
                source,
                garment,
                self.config,
                self.project_root,
                self.run_log,
                self.cancelled,
                self.status_callback,
            )
        except BaseException as error:
            self.phase = GenerationPhase.FAILED
            self._record(
                "prepare_visual_inputs",
                "failed",
                error_type=type(error).__name__,
            )
            raise
        self._prepared_inputs = prepared
        self.phase = GenerationPhase.INPUTS_PREPARED
        self._record("prepare_visual_inputs", "completed")
        return prepared

    def approve_visual_inputs(self, inputs: Any) -> str:
        self._require_phase(GenerationPhase.INPUTS_PREPARED)
        if inputs is not self._prepared_inputs:
            raise GenerationOrchestrationError(
                "오케스트레이터가 준비하지 않은 참조 입력은 승인할 수 없습니다."
            )
        self._approve_reference_run(inputs, self.config, self.request)
        approval = getattr(inputs, "approved_generation", None)
        fingerprint = getattr(approval, "fingerprint", None)
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise GenerationOrchestrationError(
                "승인 실행 fingerprint가 생성되지 않았습니다."
            )
        self.replay_bundle = self._save_replay_bundle(
            inputs, self.config, self.request, self.project_root)
        self.phase = GenerationPhase.INPUTS_APPROVED
        self._record(
            "approve_visual_inputs",
            "completed",
            approval_fingerprint=fingerprint,
            replay_bundle=str(self.replay_bundle),
        )
        return fingerprint

    def restore_approved_inputs(self, inputs: Any) -> str:
        self._require_phase(GenerationPhase.CREATED)
        approval = getattr(inputs, "approved_generation", None)
        fingerprint = getattr(approval, "fingerprint", None)
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise GenerationOrchestrationError(
                "재생 입력에 승인 fingerprint가 없습니다."
            )
        self._prepared_inputs = inputs
        self.phase = GenerationPhase.INPUTS_APPROVED
        self.require_approved_inputs(inputs)
        self._record(
            "restore_approved_inputs",
            "completed",
            approval_fingerprint=fingerprint,
        )
        return fingerprint

    def require_approved_inputs(self, inputs: Any) -> Any:
        self._require_phase(GenerationPhase.INPUTS_APPROVED)
        if inputs is not self._prepared_inputs:
            raise GenerationOrchestrationError(
                "승인 입력과 생성 입력이 서로 다릅니다."
            )
        return self._require_reference_run(
            inputs, self.config, self.request
        )



    def generate_base_candidates(self, pipeline: Any, inputs: Any) -> Any:
        approval = self.require_approved_inputs(inputs)
        from genai_lab.provenance import recorder
        model_observation = getattr(pipeline, "_run_provenance", None)
        current_observation = recorder(self.config)
        if model_observation is not None and model_observation is not current_observation:
            current_observation.snapshots.extend(model_observation.snapshots)
            for key, value in model_observation.read.items():
                current_observation.access(key, value)
            model_observation.remove_hooks()
        self.phase = GenerationPhase.BASE_RUNNING
        from genai_lab.native_pipeline_contract import final_refinement_enabled
        from genai_lab.request_runtime import reset_request_runtime

        reset_report = reset_request_runtime(
            pipeline, boundary="request_start"
        )
        self._record(
            "request_runtime_reset", "completed", report=reset_report
        )
        self.run_context.event(
            "generate_base_candidates",
            "started",
            refinement_mode=self.refinement_mode,
            runtime_reset=reset_report,
        )
        self._record(
            "generate_base_candidates",
            "started",
            refinement_mode=self.refinement_mode,
        )
        if self.refinement_mode == "sdxl_local":
            self._generation_pipeline = pipeline
        try:
            batch = self._generate_visual_batch(
                pipeline,
                self.config,
                self.request,
                self.project_root,
                self.run_log,
                inputs,
                self.cancelled,
                self.status_callback,
                defer_final_refinement=final_refinement_enabled(
                    self.config
                ),
            )
        except BaseException as error:
            failure_reset = reset_request_runtime(
                pipeline, boundary="request_failed"
            )
            self._generation_pipeline = None
            self.phase = GenerationPhase.FAILED
            self.run_context.finish(
                "failed",
                failure_stage="generate_base_candidates",
                error_type=type(error).__name__,
                runtime_reset=failure_reset,
            )
            self._record(
                "request_runtime_reset", "completed", report=failure_reset
            )
            self._record(
                "generate_base_candidates",
                "failed",
                error_type=type(error).__name__,
            )
            raise
        self.phase = GenerationPhase.BASE_COMPLETED
        self._base_batch = batch
        self.run_context.event(
            "generate_base_candidates",
            "completed",
            batch_directory=str(getattr(batch, "directory", "")),
        )
        self._record("generate_base_candidates", "completed")
        self._write_base_evidence(batch, approval)
        return batch

    def close_request(
        self,
        *,
        reason: str,
        pipeline: Any | None = None,
    ) -> dict[str, Any]:
        """Close an unfinished request and clear request-owned model state."""
        from genai_lab.request_runtime import reset_request_runtime

        if self.phase is GenerationPhase.COMPLETED:
            return {
                "version": "request_runtime_reset_v1",
                "boundary": "request_end",
                "status": "already_completed",
                "actions": [],
                "errors": [],
            }
        boundary = (
            "request_failed"
            if self.phase is GenerationPhase.FAILED
            else "request_end"
        )
        active_pipeline = self._generation_pipeline or pipeline
        report = reset_request_runtime(active_pipeline, boundary=boundary)
        self._generation_pipeline = None
        if self.phase is not GenerationPhase.FAILED:
            self.phase = GenerationPhase.CANCELLED
            self.run_context.finish(
                "cancelled",
                reason=reason,
                runtime_reset=report,
                refinement_mode=self.refinement_mode,
            )
        self._record(
            "request_runtime_reset",
            "completed",
            reason=reason,
            report=report,
        )
        self.close()
        return report

    def _write_base_evidence(
        self, batch: Any, approval: Any
    ) -> GenerationExecutionEvidence | None:
        directory = getattr(batch, "directory", None)
        if directory is None:
            return None
        review_stage = str(getattr(batch, "review_stage", "final"))
        candidate_count = len(getattr(batch, "candidates", ()))
        is_final = review_stage == "final" and candidate_count > 0
        candidates = list(getattr(batch, "candidates", ()))
        record = {
            "version": self.VERSION,
            "execution_scope": "product_pipeline",
            "runtime_smoke": False,
            "stage": "base_generation",
            "status": (
                "FINAL_GATE_PASS" if is_final else "BASE_GATE_PASS"
            ),
            "review_stage": review_stage,
            "candidate_count": candidate_count,
            "candidate_seeds": [
                getattr(candidate, "seed", None) for candidate in candidates
            ],
            "final_return_eligible": is_final,
            "approval_fingerprint": getattr(approval, "fingerprint", None),
            "base_seed": getattr(self.request, "seed", None),
            "model_id": getattr(self.request, "model_id", None),
            "replay_bundle": (
                str(self.replay_bundle)
                if self.replay_bundle is not None else None
            ),
            "trace": list(self.trace),
        }
        path = Path(directory) / self.MANIFEST_NAME
        self._write_manifest(path, record)
        return GenerationExecutionEvidence(path, record)

    def select_base_candidate(
        self,
        batch: Any,
        index: int,
    ) -> BaseCandidateSelection:
        """검증된 Base batch에서 Native 단계로 넘길 후보 하나를 봉인한다."""
        self._require_phase(GenerationPhase.BASE_COMPLETED)
        if batch is not self._base_batch:
            raise GenerationOrchestrationError(
                "현재 오케스트레이터가 생성한 Base batch가 아닙니다."
            )
        paths = list(getattr(batch, "paths", ()))
        candidates = list(getattr(batch, "candidates", ()))
        if not paths:
            raise GenerationOrchestrationError("선택할 Base 후보가 없습니다.")
        if len(paths) != len(candidates):
            raise GenerationOrchestrationError(
                "Base 후보 이미지와 진단 레코드 개수가 일치하지 않습니다."
            )
        if type(index) is not int or not 0 <= index < len(paths):
            raise GenerationOrchestrationError(
                f"Base 후보 번호 범위 오류: 1~{len(paths)}"
            )
        if str(getattr(batch, "review_stage", "")) != "native_base":
            raise GenerationOrchestrationError(
                "Native 후보 선택은 native_base 단계에서만 가능합니다."
            )
        directory = Path(batch.directory)
        source_base = Path(paths[index])
        source_record = source_base.with_suffix(".json")
        source_approved = directory / "approved_generation.json"
        source_garment = directory / "input_garment.png"
        for label, artifact in (
            ("Base 이미지", source_base),
            ("Base 후보 기록", source_record),
            ("승인 실행 기록", source_approved),
            ("격리 의상 참조", source_garment),
        ):
            if not artifact.is_file():
                raise GenerationOrchestrationError(
                    f"{label} 파일이 없습니다: {artifact}"
                )
        candidate = candidates[index]
        candidate_number = getattr(candidate, "candidate_number", index + 1)
        if type(candidate_number) is not int:
            candidate_number = index + 1
        sealed_base = self.run_context.base_directory / "approved-base.png"
        sealed_record = self.run_context.base_directory / "approved-base.json"
        sealed_approved = self.run_context.inputs_directory / "approved-generation.json"
        sealed_garment = self.run_context.inputs_directory / "input_garment.png"
        for source, target in (
            (source_base, sealed_base),
            (source_record, sealed_record),
            (source_approved, sealed_approved),
            (source_garment, sealed_garment),
        ):
            shutil.copy2(source, target)
        selection = BaseCandidateSelection(
            index=index,
            candidate_number=candidate_number,
            base_image=sealed_base,
            candidate_record=sealed_record,
            approved_run=sealed_approved,
            garment_reference=sealed_garment,
            batch_directory=directory,
        )
        self.run_context.record_artifact(
            "approved_base", sealed_base, digest=True
        )
        self.run_context.record_artifact(
            "candidate_record", sealed_record, digest=True
        )
        self.run_context.record_artifact(
            "approved_generation", sealed_approved, digest=True
        )
        self.run_context.record_artifact(
            "garment_reference", sealed_garment, digest=True
        )
        self._selected_base = selection
        self.phase = GenerationPhase.CANDIDATE_SELECTED
        self._record(
            "select_base_candidate",
            "completed",
            candidate_number=candidate_number,
            base_image=str(sealed_base),
            base_sha256=self.run_context.record["artifacts"][
                "approved_base"
            ]["sha256"],
            refinement_mode=self.refinement_mode,
        )
        self.run_context.event(
            "select_base_candidate",
            "completed",
            candidate_number=candidate_number,
            base_image=str(sealed_base),
        )
        return selection

    def finalize_selected_candidate(
        self,
        selection: BaseCandidateSelection,
        *,
        output_directory: Path | None = None,
        status_callback: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Any:
        """Finalize one approved Base with exactly one sealed refinement mode."""
        self._require_phase(GenerationPhase.CANDIDATE_SELECTED)
        if selection is not self._selected_base:
            raise GenerationOrchestrationError(
                "현재 오케스트레이터가 선택한 Base 후보가 아닙니다."
            )
        if self.refinement_mode != "sdxl_local":
            raise GenerationOrchestrationError(
                f"봉인된 정밀화 실행 모드가 없습니다: {self.refinement_mode}"
            )
        callback = status_callback or self.status_callback
        self.phase = GenerationPhase.FINAL_RUNNING
        self._record(
            "finalize_selected_candidate",
            "started",
            candidate_number=selection.candidate_number,
            refinement_mode=self.refinement_mode,
        )
        if output_directory is None:
            output_directory = self.run_context.directory_for_mode(
                self.refinement_mode
            )
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        self.run_context.event(
            "finalize_selected_candidate",
            "started",
            refinement_mode=self.refinement_mode,
            output_directory=str(output_directory),
        )
        self._record(
            "finalize_selected_candidate",
            "output_resolved",
            output_directory=str(output_directory),
        )
        result = None
        reset_report = None
        finalization_succeeded = False
        try:
            if self._generation_pipeline is None:
                raise GenerationOrchestrationError(
                    "SDXL 국소 정밀화에 Base 생성 파이프라인이 없습니다."
                )
            result = self._execute_sdxl_local_refinement_stage(
                self._generation_pipeline,
                selection=selection,
                output_directory=output_directory,
                status_callback=callback,
            )

            callback("최종 인물 수 진단 기록 중...")
            from PIL import Image
            from genai_lab.person_count_diagnostic import (
                evaluate_person_count,
                resolve_person_count_diagnostic,
            )
            person_settings = resolve_person_count_diagnostic(self.config)
            if person_settings.enabled:
                with Image.open(result.selected_image_path) as opened:
                    person_report = evaluate_person_count(
                        opened.convert("RGB"),
                        self.config,
                        output_directory=self.run_context.diagnostics_directory,
                    )
            else:
                person_report = {
                    **person_settings.record(),
                    "status": "DISABLED",
                    "action": "none",
                }
            from genai_lab.provenance import observe_gate
            observe_gate(self.config, "person_count", person_report, "final")
            result.report["person_count_diagnostic"] = person_report
            from genai_lab.final_candidate_review import (
                build_final_review_evidence,
                write_final_review_evidence,
            )
            review_evidence = build_final_review_evidence(
                result,
                selection,
                request=self.request,
            )
            review_evidence_path = write_final_review_evidence(
                review_evidence,
                output_directory,
            )
            self._final_review_evidence = review_evidence
            self._final_review_output_directory = output_directory
            result.report["final_review"] = {
                "status": "pending_user_decision",
                "evidence_path": str(review_evidence_path),
                "candidate_id": review_evidence.candidate_id,
            }
            self._write_manifest(result.report_path, result.report)
            self._write_final_product_evidence(
                result,
                selection=selection,
                output_directory=output_directory,
                person_report=person_report,
                final_review_evidence_path=review_evidence_path,
            )
            finalization_succeeded = True
        except BaseException as error:
            self.phase = GenerationPhase.FAILED
            self.run_context.finish(
                "failed",
                failure_stage="finalize_selected_candidate",
                error_type=type(error).__name__,
                refinement_mode=self.refinement_mode,
            )
            self._record(
                "finalize_selected_candidate",
                "failed",
                error_type=type(error).__name__,
                refinement_mode=self.refinement_mode,
            )
            raise
        finally:
            from genai_lab.request_runtime import reset_request_runtime
            reset_report = reset_request_runtime(
                self._generation_pipeline,
                boundary=(
                    "request_end"
                    if finalization_succeeded
                    else "request_failed"
                ),
            )
            self._record(
                "request_runtime_reset", "completed", report=reset_report
            )
            self._generation_pipeline = None
            if not finalization_succeeded:
                self.close()

        self.phase = GenerationPhase.COMPLETED
        self.run_context.finish(
            "completed",
            refinement_mode=self.refinement_mode,
            selected_engine=result.selected_engine,
            selected_image=str(result.selected_image_path),
            runtime_reset=reset_report,
        )
        self._record(
            "finalize_selected_candidate",
            "completed",
            native_status=result.status,
            selected_engine=result.selected_engine,
            selected_image=str(result.selected_image_path),
            refinement_mode=self.refinement_mode,
        )
        return result

    @property
    def final_review_evidence(self) -> Any | None:
        return self._final_review_evidence

    def record_final_review_decision(self, decision: Any) -> Path:
        """Persist one explicit Stage 8 decision after technical completion."""
        if self.phase is not GenerationPhase.COMPLETED:
            raise GenerationOrchestrationError(
                "최종 생성 완료 후에만 사용자 결정을 기록할 수 있습니다."
            )
        if self._final_review_evidence is None:
            raise GenerationOrchestrationError(
                "8단계 비교 증거가 준비되지 않았습니다."
            )
        if decision.candidate_id != self._final_review_evidence.candidate_id:
            raise GenerationOrchestrationError(
                "현재 후보와 다른 사용자 결정은 기록할 수 없습니다."
            )
        if self._final_review_output_directory is None:
            raise GenerationOrchestrationError(
                "8단계 결정 저장 경로가 없습니다."
            )
        from genai_lab.final_candidate_review import (
            write_final_review_decision,
        )
        path = write_final_review_decision(
            decision,
            self._final_review_output_directory,
        )
        self.run_context.record.setdefault("events", []).append({
            "stage": "final_candidate_review",
            "status": decision.decision,
            "candidate_id": decision.candidate_id,
            "next_action": decision.next_action,
            "decision_path": str(path),
        })
        self.run_context.record.setdefault("artifacts", {})[
            "stage8_user_decision"
        ] = {"path": str(path)}
        self.run_context.record["user_approval_status"] = decision.decision
        self.run_context.write()

        product_path = (
            self._final_review_output_directory / self.MANIFEST_NAME
        )
        if product_path.is_file():
            product = json.loads(product_path.read_text(encoding="utf-8"))
            product["user_approval_status"] = decision.decision
            product["stage8_user_decision"] = str(path)
            product["next_action"] = decision.next_action
            self._write_manifest(product_path, product)
        self._record(
            "final_candidate_review",
            decision.decision,
            candidate_id=decision.candidate_id,
            next_action=decision.next_action,
            decision_path=str(path),
        )
        return path

    def record_final_result_storage(
        self,
        storage_record: Any,
        storage_record_path: Path | str,
    ) -> None:
        """Attach the explicit Stage 9 save choice to the shared run."""
        if self.phase is not GenerationPhase.COMPLETED:
            raise GenerationOrchestrationError(
                "최종 생성 완료 후에만 저장 결정을 기록할 수 있습니다."
            )
        if self._final_review_evidence is None:
            raise GenerationOrchestrationError(
                "8단계 비교 증거가 준비되지 않았습니다."
            )
        if storage_record.candidate_id != self._final_review_evidence.candidate_id:
            raise GenerationOrchestrationError(
                "현재 후보와 다른 저장 결정은 기록할 수 없습니다."
            )
        path = Path(storage_record_path)
        self.run_context.record.setdefault("events", []).append({
            "stage": "final_result_storage",
            "status": storage_record.status,
            "candidate_id": storage_record.candidate_id,
            "storage_class": storage_record.storage_class,
            "storage_record_path": str(path),
        })
        self.run_context.record.setdefault("artifacts", {})[
            "stage9_storage_decision"
        ] = {"path": str(path)}
        self.run_context.record["user_save_status"] = storage_record.status
        if storage_record.image_path:
            self.run_context.record["saved_result"] = {
                "storage_class": storage_record.storage_class,
                "image_path": storage_record.image_path,
                "metadata_path": storage_record.metadata_path,
                "selected_output_root": storage_record.selected_output_root,
            }
        self.run_context.write()

        if self._final_review_output_directory is not None:
            product_path = (
                self._final_review_output_directory / self.MANIFEST_NAME
            )
            if product_path.is_file():
                product = json.loads(product_path.read_text(encoding="utf-8"))
                product["user_save_status"] = storage_record.status
                product["stage9_storage_decision"] = str(path)
                if storage_record.image_path:
                    product["saved_result"] = {
                        "storage_class": storage_record.storage_class,
                        "image_path": storage_record.image_path,
                        "metadata_path": storage_record.metadata_path,
                        "selected_output_root": (
                            storage_record.selected_output_root
                        ),
                    }
                self._write_manifest(product_path, product)
        self._record(
            "final_result_storage",
            storage_record.status,
            candidate_id=storage_record.candidate_id,
            storage_class=storage_record.storage_class,
            storage_record_path=str(path),
        )
        self.close()

    def _execute_sdxl_local_refinement_stage(
        self,
        pipeline: Any,
        *,
        selection: BaseCandidateSelection,
        output_directory: Path,
        status_callback: Callable[[str], None],
    ) -> Any:
        if selection.input_mode != "approved_base":
            raise GenerationOrchestrationError(
                "SDXL 국소 정밀화는 승인 Animagine Base만 입력받습니다."
            )
        from PIL import Image
        from genai_lab.refinement_result import (
            RefinementExecutionResult,
        )
        from genai_lab.selected_garment_correction import (
            correct_selected_garment,
        )
        from genai_lab.candidate_structure_gate import (
            evaluate_candidate_structure,
            resolve_candidate_structure_gate,
        )

        approved_record = json.loads(
            selection.approved_run.read_text(encoding="utf-8")
        )
        candidate_record = json.loads(
            selection.candidate_record.read_text(encoding="utf-8")
        )
        candidate_seed = int(
            candidate_record.get("seed", getattr(self.request, "seed", 0))
        )
        candidate_prompt = str(
            candidate_record.get("prompt", approved_record.get("prompt", ""))
        )
        candidate_negative = str(
            candidate_record.get(
                "negative_prompt", approved_record.get("negative_prompt", "")
            )
        )
        try:
            selected_request = replace(
                self.request,
                seed=candidate_seed,
                candidate_number=selection.candidate_number,
                prompt=candidate_prompt,
                negative_prompt=candidate_negative,
            )
        except TypeError:
            request_values = dict(vars(self.request))
            request_values.update(
                seed=candidate_seed,
                candidate_number=selection.candidate_number,
                prompt=candidate_prompt,
                negative_prompt=candidate_negative,
            )
            selected_request = SimpleNamespace(**request_values)

        source_regions = candidate_record.get(
            "output_coordinate_regions_directory"
        )
        isolated_regions = None
        if isinstance(source_regions, str) and Path(source_regions).is_dir():
            isolated_regions = self.run_context.masks_directory / "output-regions"
            shutil.copytree(
                Path(source_regions), isolated_regions, dirs_exist_ok=True
            )
        diagnostic_directory = (
            self.run_context.masks_directory / "local-refinement"
        )
        status_callback("SDXL 출력 좌표 마스크와 보호 영역 계산 중...")
        with Image.open(selection.base_image) as base_opened, Image.open(
            selection.garment_reference
        ) as garment_opened:
            base_image = base_opened.convert("RGB").copy()
            garment_image = garment_opened.convert("RGB").copy()
        visual_inputs = SimpleNamespace(garment=garment_image)
        corrected = None
        try:
            status_callback("SDXL 국소 의상 정밀화 실행 중...")
            corrected, local_report = correct_selected_garment(
                pipeline,
                base_image,
                visual_inputs,
                self.config,
                selected_request,
                self.project_root,
                approved_run_record=approved_record,
                check_running=(
                    lambda: (_ for _ in ()).throw(
                        InterruptedError("국소 정밀화를 취소했습니다.")
                    )
                    if self.cancelled()
                    else None
                ),
                run_log=self.run_log,
                output_regions_directory=isolated_regions,
                diagnostic_directory=diagnostic_directory,
                restore_pipeline_state=False,
            )
            structure_report = evaluate_candidate_structure(
                corrected, resolve_candidate_structure_gate(self.config)
            )
            from genai_lab.provenance import observe_gate
            observe_gate(self.config, "structure", structure_report, "final")
            local_report["candidate_structure_gate"] = structure_report
            local_status = str(local_report.get("status", "unresolved"))
            pass_statuses = {
                "corrected_and_accepted",
                "corrected_review_required",
            }
            accepted = (
                local_status in pass_statuses
                and structure_report.get("action") != "reject"
            )
            if accepted:
                selected_path = output_directory / "selected.png"
                corrected.save(selected_path)
                from genai_lab.provenance import observe_action
                observe_action(self.config, "structure", "final", "output_saved")
                status = "PASS"
                selected_engine = "sdxl_local"
            else:
                selected_path = selection.base_image
                from genai_lab.provenance import observe_action
                observe_action(self.config, "structure", "final", "base_returned")
                status = "BASE_LOCKED"
                selected_engine = None
            similarity = local_report.get("after_similarity")
            percentage = (
                None
                if similarity is None
                else max(0.0, min(100.0, float(similarity) * 100.0))
            )
            targets = list(local_report.get("review_reasons", ()))
            if structure_report.get("action") == "reject":
                targets.extend(structure_report.get("violations", ()))
            gate = {
                "overall_similarity_percentage": percentage,
                "refinement_required": bool(targets),
                "refinement_targets": list(dict.fromkeys(targets)),
                "hard_safety_passed": structure_report.get("action") != "reject",
            }
            observe_gate(self.config, "hard_safety",
                         {"status": "PASS" if gate["hard_safety_passed"] else "FAIL"}, "final")
            observe_action(self.config, "hard_safety", "final",
                           "output_saved" if accepted else "base_returned")
            report = {
                "version": "sdxl_local_refinement_run_v1",
                "status": status,
                "refinement_mode": "sdxl_local",
                "selected_output": str(selected_path),
                "decision": {
                    "selected_engine": selected_engine,
                    "reason": local_status,
                },
                "attempts": {"sdxl_local": {"gate": gate}},
                "local_refinement": local_report,
                "diagnostic_images": dict(
                    local_report.get("diagnostic_images", {})
                ),
                "image_merge_used": False,
                "hard_paste_used": False,
                "mask_composite_used": False,
                "approved_base_sha256": self.run_context.record[
                    "artifacts"
                ]["approved_base"]["sha256"],
            }
            report_path = output_directory / "run.json"
            self._write_manifest(report_path, report)
            return RefinementExecutionResult(
                status=status,
                selected_engine=selected_engine,
                selected_image_path=selected_path,
                report_path=report_path,
                output_directory=output_directory,
                report=report,
            )
        finally:
            if corrected is not None and corrected is not base_image:
                corrected.close()
            base_image.close()
            garment_image.close()

    def _write_final_product_evidence(
        self,
        result: Any,
        *,
        selection: BaseCandidateSelection,
        output_directory: Path,
        person_report: dict[str, Any],
        final_review_evidence_path: Path,
    ) -> None:
        record = {
            "version": self.VERSION,
            "execution_scope": "product_pipeline",
            "runtime_smoke": False,
            "stage": "final_refinement",
            "status": (
                "FINAL_GATE_PASS"
                if result.status == "PASS"
                else "FINAL_BASE_LOCKED"
            ),
            "refinement_mode": self.refinement_mode,
            "native_status": result.status,
            "input_mode": selection.input_mode,
            "selected_engine": result.selected_engine,
            "selected_image": str(result.selected_image_path),
            "final_return_eligible": True,
            "user_approval_status": "pending",
            "final_review_evidence": str(final_review_evidence_path),
            "approved_base": str(selection.base_image),
            "approved_base_sha256": self.run_context.record["artifacts"].get(
                "approved_base", {}
            ).get("sha256"),
            "source_approved_run": str(selection.approved_run),
            "refinement_report": str(result.report_path),
            "person_count_diagnostic": person_report,
            "run_context": str(self.run_context.manifest_path),
            "image_merge_used": False,
            "hard_paste_used": False,
            "mask_composite_used": False,
        }
        self._write_manifest(
            Path(output_directory) / self.MANIFEST_NAME, record
        )



    @staticmethod
    def _write_manifest(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
