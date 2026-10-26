//! The TurboCatalogService and TurboTrainingService routes (#1360).

use axum::Router;
use axum::routing::{get, post};

use crate::App;
use crate::training::{
    DeleteStaticModelRequest, InstallModelRequest, ListInstalledModelsRequest, ListModelCatalogRequest,
    ListStaticModelsRequest, ListTeachersRequest, RecordLicenceAcceptanceRequest, TrainStaticModelRequest,
};

pub(super) fn routes(r: Router<App>) -> Router<App> {
    r.route("/api/catalog/models", get(models).post(models_for))
        .route("/api/catalog/installed", get(installed))
        .route("/api/catalog/accept", post(accept))
        .route("/api/catalog/install", post(install))
        .route("/api/training/teachers", get(teachers))
        .route("/api/training/static", post(train_static))
        .route("/api/training/models", get(static_models))
        .route("/api/training/delete", post(delete_static))
}

listing!(
    /// `GET /api/catalog/models`: ListModelCatalog, every entry.
    models: catalog.list_model_catalog(ListModelCatalogRequest)
);
unary!(
    /// `POST /api/catalog/models`: ListModelCatalog for a user, by kind or
    /// language.
    models_for: catalog.list_model_catalog(ListModelCatalogRequest)
);
listing!(
    /// `GET /api/catalog/installed`: ListInstalledModels.
    installed: catalog.list_installed_models(ListInstalledModelsRequest)
);
unary!(
    /// `POST /api/catalog/accept`: RecordLicenceAcceptance, an admin call.
    accept: catalog.record_licence_acceptance(RecordLicenceAcceptanceRequest)
);
streamed!(
    /// `POST /api/catalog/install`: InstallModel, an admin call, each
    /// InstallModelUpdate as it comes.
    install: catalog.install_model(InstallModelRequest)
);
listing!(
    /// `GET /api/training/teachers`: ListTeachers.
    teachers: training.list_teachers(ListTeachersRequest)
);
streamed!(
    /// `POST /api/training/static`: TrainStaticModel, an admin call, each
    /// TrainStaticModelUpdate as it comes.
    train_static: training.train_static_model(TrainStaticModelRequest)
);
listing!(
    /// `GET /api/training/models`: ListStaticModels.
    static_models: training.list_static_models(ListStaticModelsRequest)
);
unary!(
    /// `POST /api/training/delete`: DeleteStaticModel, an admin call.
    delete_static: training.delete_static_model(DeleteStaticModelRequest)
);
