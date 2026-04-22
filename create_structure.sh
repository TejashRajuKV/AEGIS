#!/bin/bash

# ============================================================
# AEGIS - AI Fairness Autopilot System
# Full Project Folder & File Structure Builder
# ============================================================

BASE="/home/z/my-project/AEGIS-Full-Project"

# --------------------------------------------------
# ROOT LEVEL FILES
# --------------------------------------------------
touch "$BASE/README.md"
touch "$BASE/.gitignore"
touch "$BASE/.env.example"
touch "$BASE/docker-compose.yml"
touch "$BASE/Dockerfile.frontend"
touch "$BASE/Dockerfile.backend"
touch "$BASE/Makefile"
touch "$BASE/aegis-dataset-setup.py"
touch "$BASE/demo-script.md"

# --------------------------------------------------
# SHARED / DOCS
# --------------------------------------------------
mkdir -p "$BASE/aegis-shared/docs"
touch "$BASE/aegis-shared/docs/architecture-diagram.md"
touch "$BASE/aegis-shared/docs/api-specification.md"
touch "$BASE/aegis-shared/docs/ML-model-design.md"
touch "$BASE/aegis-shared/docs/deployment-guide.md"
touch "$BASE/aegis-shared/docs/demo-flow.md"
touch "$BASE/aegis-shared/docs/judge-pitch-notes.md"

mkdir -p "$BASE/aegis-shared/config"
touch "$BASE/aegis-shared/config/aegis-config.json"
touch "$BASE/aegis-shared/config/fairness-metrics.json"
touch "$BASE/aegis-shared/config/model-registry.json"
touch "$BASE/aegis-shared/config/drift-thresholds.json"

# --------------------------------------------------
# DATASETS (Preserve from original zip)
# --------------------------------------------------
mkdir -p "$BASE/aegis-shared/datasets"
touch "$BASE/aegis-shared/datasets/adult_census.csv"
touch "$BASE/aegis-shared/datasets/compas_recidivism.csv"
touch "$BASE/aegis-shared/datasets/crows_pairs.csv"
touch "$BASE/aegis-shared/datasets/electricity_drift.csv"
touch "$BASE/aegis-shared/datasets/german_credit.csv"
touch "$BASE/aegis-shared/datasets/ihdp_npci_1.csv"
touch "$BASE/aegis-shared/datasets/sachs_proteins.csv"
touch "$BASE/aegis-shared/datasets/sea_drift.csv"
touch "$BASE/aegis-shared/datasets/stereoset.csv"

# --------------------------------------------------
# FRONTEND - Next.js + React + Tailwind + Three.js
# --------------------------------------------------
FE="$BASE/aegis-frontend"

# Root config
touch "$FE/package.json"
touch "$FE/package-lock.json"
touch "$FE/next.config.js"
touch "$FE/tsconfig.json"
touch "$FE/tailwind.config.ts"
touch "$FE/postcss.config.js"
touch "$FE/.eslintrc.json"
touch "$FE/.env.local"

# Public assets
mkdir -p "$FE/public"
mkdir -p "$FE/public/fonts"
touch "$FE/public/favicon.ico"
touch "$FE/public/logo-aegis.svg"
touch "$FE/public/og-image.png"

# --------------------------------------------------
# FRONTEND - App Router (Next.js 14+)
# --------------------------------------------------
mkdir -p "$FE/src/app"
mkdir -p "$FE/src/app/dashboard"
mkdir -p "$FE/src/app/causal-graph"
mkdir -p "$FE/src/app/fairness-audit"
mkdir -p "$FE/src/app/drift-monitor"
mkdir -p "$FE/src/app/autopilot"
mkdir -p "$FE/src/app/counterfactual"
mkdir -p "$FE/src/app/text-bias"
mkdir -p "$FE/src/app/code-fix"
mkdir -p "$FE/src/app/demo"
mkdir -p "$FE/src/app/api"

touch "$FE/src/app/layout.tsx"
touch "$FE/src/app/page.tsx"
touch "$FE/src/app/globals.css"
touch "$FE/src/app/not-found.tsx"
touch "$FE/src/app/loading.tsx"

touch "$FE/src/app/dashboard/page.tsx"
touch "$FE/src/app/causal-graph/page.tsx"
touch "$FE/src/app/fairness-audit/page.tsx"
touch "$FE/src/app/drift-monitor/page.tsx"
touch "$FE/src/app/autopilot/page.tsx"
touch "$FE/src/app/counterfactual/page.tsx"
touch "$FE/src/app/text-bias/page.tsx"
touch "$FE/src/app/code-fix/page.tsx"
touch "$FE/src/app/demo/page.tsx"

# --------------------------------------------------
# FRONTEND - Components
# --------------------------------------------------
mkdir -p "$FE/src/components/layout"
touch "$FE/src/components/layout/Sidebar.tsx"
touch "$FE/src/components/layout/TopNavbar.tsx"
touch "$FE/src/components/layout/Footer.tsx"
touch "$FE/src/components/layout/MobileDrawer.tsx"
touch "$FE/src/components/layout/PageWrapper.tsx"
touch "$FE/src/components/layout/CommandPalette.tsx"

mkdir -p "$FE/src/components/ui"
touch "$FE/src/components/ui/Button.tsx"
touch "$FE/src/components/ui/Card.tsx"
touch "$FE/src/components/ui/Modal.tsx"
touch "$FE/src/components/ui/Tooltip.tsx"
touch "$FE/src/components/ui/Badge.tsx"
touch "$FE/src/components/ui/Tabs.tsx"
touch "$FE/src/components/ui/Table.tsx"
touch "$FE/src/components/ui/Input.tsx"
touch "$FE/src/components/ui/Select.tsx"
touch "$FE/src/components/ui/Slider.tsx"
touch "$FE/src/components/ui/ProgressBar.tsx"
touch "$FE/src/components/ui/Skeleton.tsx"
touch "$FE/src/components/ui/Alert.tsx"
touch "$FE/src/components/ui/Dropdown.tsx"
touch "$FE/src/components/ui/Toast.tsx"
touch "$FE/src/components/ui/CodeBlock.tsx"

mkdir -p "$FE/src/components/dashboard"
touch "$FE/src/components/dashboard/SystemHealthCard.tsx"
touch "$FE/src/components/dashboard/FairnessScoreGauge.tsx"
touch "$FE/src/components/dashboard/DriftStatusBanner.tsx"
touch "$FE/src/components/dashboard/ActiveModelPanel.tsx"
touch "$FE/src/components/dashboard/RecentAlertsFeed.tsx"
touch "$FE/src/components/dashboard/QuickActionsGrid.tsx"
touch "$FE/src/components/dashboard/MetricTrendChart.tsx"
touch "$FE/src/components/dashboard/ParetoFrontPlot.tsx"

mkdir -p "$FE/src/components/causal"
touch "$FE/src/components/causal/ForceDirectedGraph.tsx"
touch "$FE/src/components/causal/CausalNodeDetail.tsx"
touch "$FE/src/components/causal/CausalPathInspector.tsx"
touch "$FE/src/components/causal/ProxyChainHighlighter.tsx"
touch "$FE/src/components/causal/GraphControls.tsx"
touch "$FE/src/components/causal/CausalDiscoveryProgress.tsx"

mkdir -p "$FE/src/components/fairness"
touch "$FE/src/components/fairness/BiasHeatmap.tsx"
touch "$FE/src/components/fairness/FairnessMetricTable.tsx"
touch "$FE/src/components/fairness/DemographicParityChart.tsx"
touch "$FE/src/components/fairness/EqualizedOddsChart.tsx"
touch "$FE/src/components/fairness/CalibrationCurve.tsx"
touch "$FE/src/components/fairness/SubgroupAnalysisPanel.tsx"
touch "$FE/src/components/fairness/FairnessTrendLine.tsx"
touch "$FE/src/components/fairness/BiasViolationCard.tsx"

mkdir -p "$FE/src/components/drift"
touch "$FE/src/components/drift/CUSUMChart.tsx"
touch "$FE/src/components/drift/WassersteinDistChart.tsx"
touch "$FE/src/components/drift/DistributionComparison.tsx"
touch "$FE/src/components/drift/DriftTimeline.tsx"
touch "$FE/src/components/drift/FeatureDriftGrid.tsx"
touch "$FE/src/components/drift/AlertThresholdConfig.tsx"
touch "$FE/src/components/drift/LiveDataStream.tsx"

mkdir -p "$FE/src/components/autopilot"
touch "$FE/src/components/autopilot/RLControlPanel.tsx"
touch "$FE/src/components/autopilot/PPOTrainingMonitor.tsx"
touch "$FE/src/components/autopilot/RewardFunctionVisualizer.tsx"
touch "$FE/src/components/autopilot/ThresholdAdjustmentSlider.tsx"
touch "$FE/src/components/autopilot/ParetoNavigator.tsx"
touch "$FE/src/components/autopilot/AutopilotStatusIndicator.tsx"
touch "$FE/src/components/autopilot/FeatureReweightingPanel.tsx"

mkdir -p "$FE/src/components/counterfactual"
touch "$FE/src/components/counterfactual/CounterfactualGenerator.tsx"
touch "$FE/src/components/counterfactual/BeforeAfterComparison.tsx"
touch "$FE/src/components/counterfactual/CounterfactualTable.tsx"
touch "$FE/src/components/counterfactual/LatentSpaceVisualizer.tsx"
touch "$FE/src/components/counterfactual/CVAESettings.tsx"

mkdir -p "$FE/src/components/text-bias"
touch "$FE/src/components/text-bias/TextPromptInput.tsx"
touch "$FE/src/components/text-bias/EmbeddingCosineMap.tsx"
touch "$FE/src/components/text-bias/BiasScoreCard.tsx"
touch "$FE/src/components/text-bias/SentencePairComparison.tsx"
touch "$FE/src/components/text-bias/LLMSelector.tsx"
touch "$FE/src/components/text-bias/TextAuditResults.tsx"

mkdir -p "$FE/src/components/code-fix"
touch "$FE/src/components/code-fix/AutoFixPanel.tsx"
touch "$FE/src/components/code-fix/GeneratedCodeViewer.tsx"
touch "$FE/src/components/code-fix/FixDiffViewer.tsx"
touch "$FE/src/components/code-fix/CodeApplyButton.tsx"
touch "$FE/src/components/code-fix/FixHistoryTimeline.tsx"

mkdir -p "$FE/src/components/demo"
touch "$FE/src/components/demo/DemoStepIndicator.tsx"
touch "$FE/src/components/demo/DemoNarrator.tsx"
touch "$FE/src/components/demo/StepOneDataUpload.tsx"
touch "$FE/src/components/demo/StepTwoCausalDiscovery.tsx"
touch "$FE/src/components/demo/StepThreeFairnessAudit.tsx"
touch "$FE/src/components/demo/StepFourAutopilot.tsx"
touch "$FE/src/components/demo/StepFiveResults.tsx"
touch "$FE/src/components/demo/DemoTimer.tsx"

mkdir -p "$FE/src/components/threejs"
touch "$FE/src/components/threejs/ThreeScene.tsx"
touch "$FE/src/components/threejs/CausalGraph3D.tsx"
touch "$FE/src/components/threejs/LatentSpace3D.tsx"
touch "$FE/src/components/threejs/ParetoSurface3D.tsx"
touch "$FE/src/components/threejs/DriftLandscape3D.tsx"
touch "$FE/src/components/threejs/NodeGeometry.tsx"
touch "$FE/src/components/threejs/EdgeAnimation.tsx"
touch "$FE/src/components/threejs/CameraController.tsx"
touch "$FE/src/components/threejs/SceneLighting.tsx"

# --------------------------------------------------
# FRONTEND - Hooks
# --------------------------------------------------
mkdir -p "$FE/src/hooks"
touch "$FE/src/hooks/useWebSocket.ts"
touch "$FE/src/hooks/useFairnessMetrics.ts"
touch "$FE/src/hooks/useCausalGraph.ts"
touch "$FE/src/hooks/useDriftDetection.ts"
touch "$FE/src/hooks/useAutopilot.ts"
touch "$FE/src/hooks/useTextBias.ts"
touch "$FE/src/hooks/useCodeFix.ts"
touch "$FE/src/hooks/useModelRegistry.ts"
touch "$FE/src/hooks/useRealtimeStream.ts"

# --------------------------------------------------
# FRONTEND - Services / API Client
# --------------------------------------------------
mkdir -p "$FE/src/services"
touch "$FE/src/services/api-client.ts"
touch "$FE/src/services/websocket-client.ts"
touch "$FE/src/services/causal-service.ts"
touch "$FE/src/services/fairness-service.ts"
touch "$FE/src/services/drift-service.ts"
touch "$FE/src/services/autopilot-service.ts"
touch "$FE/src/services/counterfactual-service.ts"
touch "$FE/src/services/text-bias-service.ts"
touch "$FE/src/services/code-fix-service.ts"
touch "$FE/src/services/model-service.ts"
touch "$FE/src/services/dataset-service.ts"

# --------------------------------------------------
# FRONTEND - State Management
# --------------------------------------------------
mkdir -p "$FE/src/store"
touch "$FE/src/store/fairness-store.ts"
touch "$FE/src/store/causal-store.ts"
touch "$FE/src/store/drift-store.ts"
touch "$FE/src/store/autopilot-store.ts"
touch "$FE/src/store/ui-store.ts"
touch "$FE/src/store/model-store.ts"
touch "$FE/src/store/demo-store.ts"

# --------------------------------------------------
# FRONTEND - Types
# --------------------------------------------------
mkdir -p "$FE/src/types"
touch "$FE/src/types/fairness.types.ts"
touch "$FE/src/types/causal.types.ts"
touch "$FE/src/types/drift.types.ts"
touch "$FE/src/types/autopilot.types.ts"
touch "$FE/src/types/counterfactual.types.ts"
touch "$FE/src/types/text-bias.types.ts"
touch "$FE/src/types/code-fix.types.ts"
touch "$FE/src/types/model.types.ts"
touch "$FE/src/types/dataset.types.ts"
touch "$FE/src/types/api.types.ts"
touch "$FE/src/types/websocket.types.ts"

# --------------------------------------------------
# FRONTEND - Utils
# --------------------------------------------------
mkdir -p "$FE/src/utils"
touch "$FE/src/utils/color-scales.ts"
touch "$FE/src/utils/graph-layout.ts"
touch "$FE/src/utils/data-transformers.ts"
touch "$FE/src/utils/formatters.ts"
touch "$FE/src/utils/constants.ts"
touch "$FE/src/utils/cn.ts"

# --------------------------------------------------
# FRONTEND - Styles
# --------------------------------------------------
mkdir -p "$FE/src/styles"
touch "$FE/src/styles/aegis-theme.css"
touch "$FE/src/styles/animations.css"
touch "$FE/src/styles/custom-scrollbars.css"
touch "$FE/src/styles/three-overrides.css"

# --------------------------------------------------
# BACKEND - Python FastAPI
# --------------------------------------------------
BE="$BASE/aegis-backend"

touch "$BE/requirements.txt"
touch "$BE/pyproject.toml"
touch "$BE/setup.py"
touch "$BE/Pipfile"
touch "$BE/.env"
touch "$BE/.env.example"
touch "$BE/.python-version"
touch "$BE/pytest.ini"
touch "$BE/run.py"
touch "$BE/manage.py"

# --------------------------------------------------
# BACKEND - App Structure
# --------------------------------------------------
mkdir -p "$BE/app"
touch "$BE/app/__init__.py"
touch "$BE/app/main.py"
touch "$BE/app/config.py"
touch "$BE/app/dependencies.py"
touch "$BE/app/exceptions.py"
touch "$BE/app/middleware.py"
touch "$BE/app/events.py"

# --------------------------------------------------
# BACKEND - API Routes
# --------------------------------------------------
mkdir -p "$BE/app/api"
touch "$BE/app/api/__init__.py"
touch "$BE/app/api/router.py"

mkdir -p "$BE/app/api/routes"
touch "$BE/app/api/routes/__init__.py"
touch "$BE/app/api/routes/health.py"
touch "$BE/app/api/routes/datasets.py"
touch "$BE/app/api/routes/causal.py"
touch "$BE/app/api/routes/fairness.py"
touch "$BE/app/api/routes/drift.py"
touch "$BE/app/api/routes/autopilot.py"
touch "$BE/app/api/routes/counterfactual.py"
touch "$BE/app/api/routes/text_bias.py"
touch "$BE/app/api/routes/code_fix.py"
touch "$BE/app/api/routes/models.py"
touch "$BE/app/api/routes/websocket.py"

# --------------------------------------------------
# BACKEND - ML Core (Person A)
# --------------------------------------------------
mkdir -p "$BE/app/ml/causal"
touch "$BE/app/ml/causal/__init__.py"
touch "$BE/app/ml/causal/dag_gnn.py"
touch "$BE/app/ml/causal/dag_gnn_model.py"
touch "$BE/app/ml/causal/pc_algorithm.py"
touch "$BE/app/ml/causal/causal_scoring.py"
touch "$BE/app/ml/causal/proxy_chain_detector.py"
touch "$BE/app/ml/causal/graph_utils.py"

mkdir -p "$BE/app/ml/fairness"
touch "$BE/app/ml/fairness/__init__.py"
touch "$BE/app/ml/fairness/metrics.py"
touch "$BE/app/ml/fairness/demographic_parity.py"
touch "$BE/app/ml/fairness/equalized_odds.py"
touch "$BE/app/ml/fairness/calibration.py"
touch "$BE/app/ml/fairness/subgroup_analysis.py"
touch "$BE/app/ml/fairness/bias_reporter.py"
touch "$BE/app/ml/fairness/fairness_pipeline.py"

mkdir -p "$BE/app/ml/rl"
touch "$BE/app/ml/rl/__init__.py"
touch "$BE/app/ml/rl/ppo_agent.py"
touch "$BE/app/ml/rl/ppo_network.py"
touch "$BE/app/ml/rl/reward_shaper.py"
touch "$BE/app/ml/rl/pareto_reward.py"
touch "$BE/app/ml/rl/action_space.py"
touch "$BE/app/ml/rl/environment.py"
touch "$BE/app/ml/rl/training_loop.py"
touch "$BE/app/ml/rl/goodhart_guard.py"

mkdir -p "$BE/app/ml/text_bias"
touch "$BE/app/ml/text_bias/__init__.py"
touch "$BE/app/ml/text_bias/llm_wrapper.py"
touch "$BE/app/ml/text_bias/embedding_extractor.py"
touch "$BE/app/ml/text_bias/cosine_distance.py"
touch "$BE/app/ml/text_bias/prompt_framer.py"
touch "$BE/app/ml/text_bias/text_auditor.py"
touch "$BE/app/ml/text_bias/bias_scorer.py"

# --------------------------------------------------
# BACKEND - Neural Networks (Person B)
# --------------------------------------------------
mkdir -p "$BE/app/ml/neural"
touch "$BE/app/ml/neural/__init__.py"
touch "$BE/app/ml/neural/conditional_vae.py"
touch "$BE/app/ml/neural/vae_encoder.py"
touch "$BE/app/ml/neural/vae_decoder.py"
touch "$BE/app/ml/neural/counterfactual_generator.py"
touch "$BE/app/ml/neural/latent_interpolator.py"
touch "$BE/app/ml/neural/vae_trainer.py"

mkdir -p "$BE/app/ml/gnn"
touch "$BE/app/ml/gnn/__init__.py"
touch "$BE/app/ml/gnn/dag_gnn_layers.py"
touch "$BE/app/ml/gnn/node_encoder.py"
touch "$BE/app/ml/gnn/edge_decoder.py"
touch "$BE/app/ml/gnn/graph_attention.py"
touch "$BE/app/ml/gnn/causal_gnn_trainer.py"

mkdir -p "$BE/app/ml/drift"
touch "$BE/app/ml/drift/__init__.py"
touch "$BE/app/ml/drift/cusum_detector.py"
touch "$BE/app/ml/drift/wasserstein_detector.py"
touch "$BE/app/ml/drift/drift_ensemble.py"
touch "$BE/app/ml/drift/temporal_window.py"
touch "$BE/app/ml/drift/distribution_comparator.py"
touch "$BE/app/ml/drift/drift_alert.py"

# --------------------------------------------------
# BACKEND - Systems Layer (Person C)
# --------------------------------------------------
mkdir -p "$BE/app/services"
touch "$BE/app/services/__init__.py"
touch "$BE/app/services/model_wrapper.py"
touch "$BE/app/services/model_registry.py"
touch "$BE/app/services/llm_client.py"
touch "$BE/app/services/auto_fix_generator.py"
touch "$BE/app/services/code_formatter.py"
touch "$BE/app/services/websocket_manager.py"
touch "$BE/app/services/task_queue.py"
touch "$BE/app/services/cache.py"
touch "$BE/app/services/file_handler.py"

mkdir -p "$BE/app/services/wrappers"
touch "$BE/app/services/wrappers/__init__.py"
touch "$BE/app/services/wrappers/sklearn_wrapper.py"
touch "$BE/app/services/wrappers/xgboost_wrapper.py"
touch "$BE/app/services/wrappers/pytorch_wrapper.py"
touch "$BE/app/services/wrappers/tensorflow_wrapper.py"
touch "$BE/app/services/wrappers/base_wrapper.py"

# --------------------------------------------------
# BACKEND - Data Layer
# --------------------------------------------------
mkdir -p "$BE/app/data"
touch "$BE/app/data/__init__.py"
touch "$BE/app/data/dataset_loader.py"
touch "$BE/app/data/preprocessor.py"
touch "$BE/app/data/feature_engineering.py"
touch "$BE/app/data/schema_validator.py"
touch "$BE/app/data/data_splitter.py"

mkdir -p "$BE/app/data/schemas"
touch "$BE/app/data/schemas/__init__.py"
touch "$BE/app/data/schemas/adult_census.py"
touch "$BE/app/data/schemas/compas.py"
touch "$BE/app/data/schemas/german_credit.py"

# --------------------------------------------------
# BACKEND - Core ML Pipeline
# --------------------------------------------------
mkdir -p "$BE/app/pipeline"
touch "$BE/app/pipeline/__init__.py"
touch "$BE/app/pipeline/audit_pipeline.py"
touch "$BE/app/pipeline/discovery_pipeline.py"
touch "$BE/app/pipeline/autopilot_pipeline.py"
touch "$BE/app/pipeline/drift_pipeline.py"
touch "$BE/app/pipeline/pipeline_coordinator.py"
touch "$BE/app/pipeline/results_aggregator.py"

# --------------------------------------------------
# BACKEND - Models DB / ORM
# --------------------------------------------------
mkdir -p "$BE/app/models"
touch "$BE/app/models/__init__.py"
touch "$BE/app/models/schemas.py"
touch "$BE/app/models/database.py"
touch "$BE/app/models/audit_record.py"
touch "$BE/app/models/model_record.py"
touch "$BE/app/models/drift_record.py"
touch "$BE/app/models/session.py"

# --------------------------------------------------
# BACKEND - Utils
# --------------------------------------------------
mkdir -p "$BE/app/utils"
touch "$BE/app/utils/__init__.py"
touch "$BE/app/utils/logger.py"
touch "$BE/app/utils/metrics_utils.py"
touch "$BE/app/utils/math_utils.py"
touch "$BE/app/utils/file_utils.py"
touch "$BE/app/utils/validation.py"

# --------------------------------------------------
# BACKEND - Tests
# --------------------------------------------------
mkdir -p "$BE/tests"
touch "$BE/tests/__init__.py"
touch "$BE/tests/conftest.py"
touch "$BE/tests/test_health.py"
touch "$BE/tests/test_causal.py"
touch "$BE/tests/test_fairness.py"
touch "$BE/tests/test_drift.py"
touch "$BE/tests/test_autopilot.py"
touch "$BE/tests/test_text_bias.py"
touch "$BE/tests/test_code_fix.py"
touch "$BE/tests/test_counterfactual.py"
touch "$BE/tests/test_pipeline.py"

mkdir -p "$BE/tests/fixtures"
touch "$BE/tests/fixtures/sample_data.py"
touch "$BE/tests/fixtures/mock_models.py"
touch "$BE/tests/fixtures/sample_graphs.py"

# --------------------------------------------------
# BACKEND - Notebooks / Research
# --------------------------------------------------
mkdir -p "$BE/notebooks"
touch "$BE/notebooks/01_causal_discovery_experiment.ipynb"
touch "$BE/notebooks/02_fairness_benchmark.ipynb"
touch "$BE/notebooks/03_rl_autopilot_training.ipynb"
touch "$BE/notebooks/04_drift_detection_analysis.ipynb"
touch "$BE/notebooks/05_cvae_counterfactual.ipynb"
touch "$BE/notebooks/06_text_bias_audit.ipynb"

# --------------------------------------------------
# BACKEND - ML Model Checkpoints
# --------------------------------------------------
mkdir -p "$BE/checkpoints"
mkdir -p "$BE/checkpoints/dag_gnn"
mkdir -p "$BE/checkpoints/conditional_vae"
mkdir -p "$BE/checkpoints/ppo_agent"
touch "$BE/checkpoints/.gitkeep"

# --------------------------------------------------
# BACKEND - Logs
# --------------------------------------------------
mkdir -p "$BE/logs"
touch "$BE/logs/.gitkeep"

# --------------------------------------------------
# BACKEND - Scripts
# --------------------------------------------------
mkdir -p "$BE/scripts"
touch "$BE/scripts/train_dag_gnn.py"
touch "$BE/scripts/train_cvae.py"
touch "$BE/scripts/train_ppo.py"
touch "$BE/scripts/run_full_audit.py"
touch "$BE/scripts/run_drift_monitor.py"
touch "$BE/scripts/generate_demo_data.py"
touch "$BE/scripts/export_results.py"

# --------------------------------------------------
# BACKEND - Static / Uploads
# --------------------------------------------------
mkdir -p "$BE/static"
touch "$BE/static/.gitkeep"
mkdir -p "$BE/uploads"
touch "$BE/uploads/.gitkeep"

echo "✅ Full AEGIS project structure created successfully!"
echo ""
echo "Total files and directories created:"
find "$BASE" -type f | wc -l
find "$BASE" -type d | wc -l
