import json
from pathlib import Path
from typing import Dict, Any, Optional

from plwordnet_handler.utils.logger import prepare_logger


class BiEncoderModelConfig:
    REQUIRED_FIELDS = ["model_name", "model_path", "vec_size"]

    def __init__(
        self,
        model_config_path: str,
        auto_load: bool = False,
        log_level: str = "INFO",
        log_filename: Optional[str] = None,
    ):
        self.model_config_path = model_config_path

        # Required fields
        self.model_name = None
        self.model_path = None
        self.vec_size = None

        # Optional fields
        self.wandb_url = None
        self.comment = None

        self.active_model_id = None
        self._config_data = None

        self.logger = prepare_logger(
            logger_name=__name__,
            logger_file_name=log_filename,
            log_level=log_level,
            use_default_config=True,
        )

        if auto_load:
            self.load()

    @classmethod
    def from_json_file(cls, config_path: str) -> "BiEncoderModelConfig":
        """
        Create a BiEncoderModelConfig instance from a JSON configuration file.

        Args:
            config_path: Path to the JSON configuration file

        Returns:
            BiEncoderModelConfig: Instance with loaded configuration

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the JSON is invalid
            KeyError: If required configuration keys are missing
            ValueError: If active model is not found in models or missing required fields
        """
        config = cls(config_path)
        config.load()
        return config

    def load(self) -> bool:
        """
        Load configuration from the JSON file and set active model properties.

        Returns:
            bool: True if loading is successful, False otherwise

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the JSON is invalid
            KeyError: If required configuration keys are missing
            ValueError: If active model is not found in models
            or missing required fields
        """
        config_path = Path(self.model_config_path)
        if not config_path.exists():
            error_msg = f"Configuration file not found: {config_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        return self._load_config(config_path=config_path)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert current active model configuration to dictionary.

        Returns:
            Dict containing current active model configuration
        """
        return {
            "active_model_id": self.active_model_id,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "vec_size": self.vec_size,
            "wandb_url": self.wandb_url,
            "comment": self.comment,
        }

    def _load_config(self, config_path: Path) -> bool:
        """
        Loads and parses JSON configuration file from disk.

        This private method handles the core file loading and JSON parsing
        operations. It reads the configuration file, validates JSON format,
        and delegates further processing to validation and model loading methods.

        Args:
            config_path: Path object pointing to the JSON configuration file

        Returns:
            bool: True if configuration loading and model setup successful

        Raises:
            json.JSONDecodeError: If the JSON file contains invalid syntax

        Side effects:
            - Sets self._config_data to the parsed JSON content
            - Trigger
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config_data = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in configuration file {config_path}: {e}"
            self.logger.error(error_msg)
            raise json.JSONDecodeError(
                f"Invalid JSON in configuration file: {e}", "", 0
            )

        return self._load_active_model() if self._check_config() else False

    def _check_config(self):
        """
        Validates the structure and required keys of loaded configuration data.

        This method performs structural validation on the loaded JSON configuration
        to ensure all required top-level keys are present and that the specified
        active model exists within the models section. It serves as a gate-keeper
        before attempting to load model-specific configuration.

        Returns:
            bool: True if all validation checks pass

        Raises:
            KeyError: If required keys 'active' or 'models' are missing
            ValueError: If the active model ID is not found in a models section

        Side effects:
            - Logs error messages for any validation failures
            - Does not modify instance state, only validates existing data
        """

        if "active" not in self._config_data:
            error_msg = "Missing required key 'active' in configuration"
            self.logger.error(error_msg)
            raise KeyError(error_msg)

        if "models" not in self._config_data:
            error_msg = "Missing required key 'models' in configuration"
            self.logger.error(error_msg)
            raise KeyError(error_msg)

        active_id = self._config_data["active"]
        models = self._config_data["models"]
        if active_id not in models:
            error_msg = f"Active model '{active_id}' not found in models section"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        return True

    def _load_active_model(self, active_id: Optional[str] = None) -> bool:
        """
        Loads and validates the active model configuration into instance attributes.

        This private method extracts the active model's configuration data,
        validates that all required fields are present, and populates the
        instance attributes with the model's properties. It ensures data
        integrity by checking for mandatory fields before assignment.

        Args:
            active_id: Optional model ID to load. If None, uses the active
                      model specified in configuration data

        Returns:
            bool: True if model loading and validation are successful

        Raises:
            ValueError: If required fields are missing from the active model

        Side effects:
            - Sets all instance attributes for the active model:
              active_model_id, model_name, model_path, vec_size,
              wandb_url, comment
            - Logs successful model loading with model details
        """

        if active_id is None:
            active_id = self._config_data["active"]

        models = self._config_data["models"]
        active_model = models[active_id]

        missing_fields = [
            field for field in self.REQUIRED_FIELDS if field not in active_model
        ]
        if missing_fields:
            error_msg = (
                f"Missing required fields in active model "
                f"'{active_id}': {missing_fields}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.active_model_id = active_id
        self.model_name = active_model["model_name"]
        self.model_path = active_model["model_path"]
        self.vec_size = active_model["vec_size"]
        self.wandb_url = active_model.get("wandb_url")
        self.comment = active_model.get("comment")

        self.logger.info(
            f"Loaded active model configuration: '{active_id}' - {self.model_name}"
        )
        return True


class BiEncoderModelConfigHandler(BiEncoderModelConfig):
    """
    Extended configuration handler for bi-encoder model management
    with enhanced operations.

    This class extends the base BiEncoderModelConfig functionality to provide
    a comprehensive handler interface for managing multiple bi-encoder model
    configurations. It serves as a higher-level abstraction that facilitates
    dynamic model switching, configuration querying, and model discovery
    operations within a multimodel environment.

    The handler is designed for scenarios where applications need to work with
    multiple pre-trained bi-encoder models and switch between them dynamically
    at runtime. It maintains backward compatibility with the base configuration
    class while adding operational methods for model management workflows.

    Key Features:
        - **Dynamic Model Switching**: Runtime switching between different
        model configurations without requiring configuration file reloading
        - **Model Discovery**: Enumeration of all available models
        with their metadata
        - **Configuration Querying**: Access to any model's configuration
        data, not just the active one
        - **Safe Model Validation**: Comprehensive validation before model
        activation to prevent runtime errors
        - **Enhanced Error Handling**: Detailed logging and graceful error
        recovery for operational robustness

    Inheritance:
        Inherits all functionality from BiEncoderModelConfig including
        - JSON configuration file loading and parsing
        - Required field validation (model_name, model_path, vec_size)
        - Optional field handling (wandb_url, comment)
        - Configuration structure validation
        - Error handling with logging

    Example:
        ```python
        # Initialize handler with configuration
        handler = BiEncoderModelConfigHandler.from_json_file("embedder-config.json")

        # Discover available models
        models = handler.get_available_models()
        print(f"Available models: {list(models.keys())}")

        # Switch to a different model
        if handler.set_active_model("alternative_model"):
            print(f"Switched to: {handler.model_name}")

        # Query specific model configuration
        model_config = handler.get_model_config("another_model")
        if model_config:
            print(f"Vector size: {model_config.get('vec_size')}")
        ```

    Configuration Format:
        Expects the same JSON format as BiEncoderModelConfig with "active"
        and "models" sections, where each model must contain the required
        fields: {model_name, model_path, and vec_size}.
    """

    def get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves configuration dictionary for a specified model by its identifier.

        This method provides access to the raw configuration data for any model
        defined in the configuration file, not just the currently active model.
        It performs a safe lookup operation that handles cases where the
        configuration data hasn't been loaded or the requested model doesn't exist.

        Args:
            model_id: Unique identifier string for the model whose
            configuration should be retrieved

        Returns:
            Dict containing the complete model configuration including all
            required and optional fields, or None if the model is not found or
            configuration is not loaded

        Side effects:
            - Logs warning message if configuration data is not yet loaded
            - Does not modify any instance state
        """
        if self._config_data is None:
            self.logger.warning("Configuration not loaded. Call load() first.")
            return None

        models = self._config_data.get("models", {})
        return models.get(model_id)

    def get_available_models(self) -> Dict[str, str]:
        """
        Compiles a mapping of all available model identifiers to their display names.

        This method scans through all models defined in the configuration file
        and creates a convenient lookup dictionary that maps model IDs to their
        human-readable names. This is useful for presenting model options to
        users or for programmatic iteration over available models.

        Returns:
            Dict mapping model ID strings to model name strings. If a model
            doesn't have a 'model_name' field, the ID is used as the name.
            Returns empty dict if the configuration is not loaded.

        Side effects:
            - Logs warning message if configuration data is not yet loaded
            - Does not modify any instance state
        """
        if self._config_data is None:
            self.logger.warning("Configuration not loaded. Call load() first.")
            return {}

        models = self._config_data.get("models", {})
        return {
            model_id: config.get("model_name", model_id)
            for model_id, config in models.items()
        }

    def set_active_model(self, model_id: str) -> bool:
        """
        Changes the currently active model to a different model from configuration.

        This method allows dynamic switching between different model configurations
        at runtime. It validates that the target model exists and has all required
        fields before making the switch. Upon successful validation, it updates
        all instance attributes to reflect the new active model's configuration.

        Args:
            model_id: Identifier string of the model to make active

        Returns:
            bool: True if the model switch was successful and all validation
                 checks passed, False if the model was not found or configuration
                 not loaded

        Raises:
            ValueError: If the target model is missing any required fields
                       (model_name, model_path, vec_size)

        Side effects:
            - Updates all active model instance attributes if successful
            - Logs error messages for validation failures
            - Logs success message with new active model details via _load_active_model
        """
        if self._config_data is None:
            self.logger.error("Configuration not loaded. Call load() first.")
            return False

        models = self._config_data.get("models", {})
        if model_id not in models:
            error_msg = f"Model '{model_id}' not found in configuration"
            self.logger.error(error_msg)
            return False
        return self._load_active_model(active_id=model_id)
