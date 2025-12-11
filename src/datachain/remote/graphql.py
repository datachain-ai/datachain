TRIGGER_DATASET_DEPENDENCY_UPDATE_MUTATION = """
mutation TriggerDatasetDependencyUpdate(
  $teamName: String!,
  $namespaceName: String!,
  $projectName: String!,
  $datasetName: String!,
  $version: String!,
  $review: Boolean
) {
  triggerDatasetDependencyUpdate(
    teamName: $teamName
    namespaceName: $namespaceName
    projectName: $projectName
    datasetName: $datasetName
    version: $version
    review: $review
  ) {
    ok
    pipeline {
      pipelineId
      status
      name
      triggeredFrom
      errorMessage
    }
  }
}
"""
