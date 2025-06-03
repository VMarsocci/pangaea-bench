from terratorch.registry import BACKBONE_REGISTRY
from pangaea.encoders.base import Encoder # Ensure this import path is correct based on your project structure
from pathlib import Path
from logging import Logger
import torch # For type hinting

# --- Placeholder Default Values ---
# These values are highly dependent on the "terramind_v1_base_tim" model's architecture.
# Please verify and adjust them according to the model's specifications and your requirements.
DEFAULT_INPUT_SIZE = 224  # Example: Common input size for vision models
DEFAULT_EMBED_DIM = 768   # Example: Common embedding dimension for ViT-Base models
DEFAULT_OUTPUT_LAYERS = [-1] # Example: Output from the last layer
# DEFAULT_OUTPUT_DIM should match the feature dimension of the layers specified in DEFAULT_OUTPUT_LAYERS
# If DEFAULT_OUTPUT_LAYERS = [-1], then DEFAULT_OUTPUT_DIM would typically be DEFAULT_EMBED_DIM.
DEFAULT_OUTPUT_DIM = DEFAULT_EMBED_DIM
# --- End Placeholder Default Values ---

class TerraMindAllEncoder(Encoder):
    """
    Encoder class for TerraMind models loaded via terratorch.registry.
    Wraps a TerraMind model (e.g., "terramind_v1_base_tim") to be compatible with
    the pangaea Encoder interface.
    """
    def __init__(
        self,
        # Parameters from Encoder base class:
        encoder_weights: str | Path = "terramind_v1_base_tim_pretrained_via_registry", # Descriptive name
        input_bands: dict[str, list[str]] = None, # Must be provided by the user
        input_size: int = DEFAULT_INPUT_SIZE,
        embed_dim: int = DEFAULT_EMBED_DIM, # Main feature dimension of the underlying TerraMind model
        output_layers: list[int] = None,    # Which layers to output features from (e.g., [-1] for last)
        output_dim: int | list[int] = None, # Dimension(s) of the output features
        download_url: str = None, # Assuming terratorch handles downloads if pretrained=True

        # Encoder behavior flags:
        multi_temporal: bool = False, # Adjust if terramind_v1_base_tim handles multi-temporal data
        multi_temporal_output: bool = False, # Adjust accordingly
        pyramid_output: bool = False, # Adjust if terramind_v1_base_tim provides pyramid features

        # Specific TerraMind model parameters (passed to BACKBONE_REGISTRY.build):
        terramind_model_name: str = "terramind_v1_base_tim",
        terramind_modalities: list[str] = None, # Defaults to ["S2L1C", "S1GRD"]
        terramind_tim_modalities: list[str] = None, # Defaults to ["LULC"]
        terramind_pretrained: bool = True,
        pangaea_modalities: list[str] = None, # Defaults to ["optical", "sar"]
    ):
        
        super().__init__(
            model_name="terramind_all_wrapper", # Custom name for this encoder wrapper
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=multi_temporal,
            multi_temporal_output=multi_temporal_output,
            pyramid_output=pyramid_output,
            encoder_weights=str(encoder_weights), # Ensure it's a string if Path object
            download_url=download_url,
        )
        
        if terramind_modalities is None:
            terramind_modalities = ["S2L1C", "S1GRD"]
        if terramind_tim_modalities is None:
            terramind_tim_modalities = ["LULC"]
        if pangaea_modalities is None:
            pangaea_modalities = ["optical", "sar"]

        # --- Input Bands Configuration ---
        # User must provide `input_bands` that map to `pangaea_modalities`.
        # Example: {"optical": ["B02", "B03", ...], "sar": ["VV", "VH"]}
        if input_bands is None:
            # Create a default structure based on pangaea_modalities.
            # The user *must* fill in the actual band names for each modality.
            input_bands = {mod: [] for mod in pangaea_modalities}
            # Example print to remind user, consider raising an error or more robust handling
            print(f"Warning: `input_bands` not provided for TerraMindAllEncoder. "
                  f"Initialized with empty band lists for modalities: {pangaea_modalities}. "
                  f"Please provide actual band names.")

        # --- Output Configuration ---
        if output_layers is None:
            output_layers = list(DEFAULT_OUTPUT_LAYERS) # Ensure it's a list copy
        
        if output_dim is None:
            # Base class handles replicating int output_dim if multiple output_layers.
            # If output_layers has multiple values, output_dim might need to be a list.
            output_dim = embed_dim # Simplest assumption: output dim matches embed_dim

        # Store terratorch model configuration
        self.terramind_model_name = terramind_model_name
        self.terramind_modalities = terramind_modalities # From config, e.g., ["S2L1C", "S1GRD"]
        self.terramind_tim_modalities = terramind_tim_modalities
        self.terramind_pretrained = terramind_pretrained
        self.pangaea_modalities = pangaea_modalities # From config, e.g., ["optical", "sar"]

        # Arguments for BACKBONE_REGISTRY.build
        build_args = {}
        build_args["pretrained"] = self.terramind_pretrained
        # build_args["use_chain_of_thoughts"] = True # Enable Chain of Thoughts as requested

        # Create mapping from Pangaea modalities to TerraMind modalities
        # Pangaea modalities are the keys expected in `input_bands` and in the `forward` method's input `x`
        # TerraMind modalities are what the terratorch model expects
        self.pangaea_to_terratorch_mod_map = dict(zip(self.pangaea_modalities, self.terramind_modalities))
        
        print(f"Pangaea modalities (keys for `input_bands`): {self.pangaea_modalities}")
        print(f"TerraMind modalities (for terratorch build): {self.terramind_modalities}")
        print(f"Pangaea-to-TerraMind modality map: {self.pangaea_to_terratorch_mod_map}")

        terratorch_modalities_for_build = list(self.terramind_modalities)
        tim_modalities_for_build = list(self.terramind_tim_modalities) if self.terramind_tim_modalities is not None else None

        # Removed attempt to pass 'out_indices' to the terratorch model constructor
        # as it caused: TypeError: TerraMindViT.__init__() got an unexpected keyword argument 'out_indices'
        # The following block was removed/commented out:
        # if self.output_layers is not None:
        #     # Ensure self.output_layers (from base Encoder, from YAML) is a Python list
        #     out_indices_for_build = list(self.output_layers)
        #     build_args["out_indices"] = out_indices_for_build 
        #     print(f"  Attempting to pass `out_indices={out_indices_for_build}` for terratorch build.")
        # else:
        #     print("  `output_layers` (for out_indices) not configured in encoder, will not be passed to terratorch build.")


        if self.terramind_model_name and isinstance(self.terramind_model_name, str):
            model_name_lower = self.terramind_model_name.lower()
            if "tim" in model_name_lower:
                print(f"Info: Detected TiM model ('{self.terramind_model_name}').")
                build_args["modalities"] = terratorch_modalities_for_build 
                print(f"  Setting `modalities=None` for terratorch build.")
                if tim_modalities_for_build:
                    build_args["tim_modalities"] = tim_modalities_for_build
                    print(f"  Adding `tim_modalities={build_args['tim_modalities']}` for terratorch build.")
                else:
                    print(f"  `terramind_tim_modalities` not configured, will not be passed.")
            else:  # Non-TiM model
                print(f"Info: Detected non-TiM model ('{self.terramind_model_name}').")
                build_args["modalities"] = terratorch_modalities_for_build 
                print(f"  Setting `modalities={build_args['modalities']}` (mapped from pangea modalities) for terratorch build.")
                if tim_modalities_for_build:
                    print(f"  Warning: `terramind_tim_modalities` ({tim_modalities_for_build}) configured but model is not TiM. Will NOT be passed.")
        else:
            print(f"Warning: `terramind_model_name` ('{self.terramind_model_name}') invalid. Using mapped modalities as fallback.")
            build_args["modalities"] = terratorch_modalities_for_build
            if tim_modalities_for_build:
                build_args["tim_modalities"] = tim_modalities_for_build # May cause issues

        print("BUILD ARGS: ", build_args)
        self.model = BACKBONE_REGISTRY.build(
            self.terramind_model_name,
            **build_args
        )

    def forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor] | torch.Tensor:
        # x input from Pangea dataloader, keys are pangaea_modalities (e.g., {"optical": tensor})
        # Map to terratorch model's expected keys (e.g., {"S2L1C": tensor})
        
        model_input = {}
        for pangea_key, tensor_val in x.items():
            if pangea_key in self.pangaea_to_terratorch_mod_map:
                terratorch_key = self.pangaea_to_terratorch_mod_map[pangea_key]
                model_input[terratorch_key] = tensor_val
            else:
                # This case should ideally not happen if pangaea_to_terratorch_mod_map is comprehensive
                print(f"Warning: No terratorch mapping for Pangea modality '{pangea_key}'. Passing directly.")
                model_input[pangea_key] = tensor_val
        
        # print(f"TerraMindAllEncoder forward: input keys {list(x.keys())} -> model_input keys {list(model_input.keys())}")

        features = None # Initialize features

        # Optional: Add these prints for debugging if the issue persists after the fix.
        # print(f"DEBUG: TerraMindAllEncoder.forward: self.model type = {type(self.model)}")
        # print(f"DEBUG: TerraMindAllEncoder.forward: hasattr(self.model, 'forward_features') = {hasattr(self.model, 'forward_features')}")
        # print(f"DEBUG: TerraMindAllEncoder.forward: self.output_layers = {self.output_layers}")

        if hasattr(self.model, 'forward_features'):
            if self.output_layers is not None and len(self.output_layers) > 0:
                # print(f"  Attempting to call self.model.forward_features with argument: output_layers={list(self.output_layers)}")
                try:
                    # CRITICAL CHANGE: Use 'output_layers' as the keyword argument name
                    features = self.model.forward_features(model_input, output_layers=list(self.output_layers))
                    # num_feats = len(features) if isinstance(features, list) else (1 if features is not None else 0)
                    # print(f"  Successfully called forward_features with output_layers. Got {num_feats} feature map(s).")
                except TypeError as e:
                    print(f"  Call to forward_features with output_layers={list(self.output_layers)} failed: {e}.")
                    print(f"  Falling back to call forward_features without output_layers argument.")
                    features = self.model.forward_features(model_input) # Fallback
            else:
                # print("  Calling forward_features without output_layers (self.output_layers is None or empty).")
                features = self.model.forward_features(model_input)
        elif hasattr(self.model, 'forward'):
            # print("  Model does not have 'forward_features', using 'forward'.")
            raw_output = self.model(model_input) 
            if isinstance(raw_output, torch.Tensor): features = [raw_output]
            elif isinstance(raw_output, list): features = raw_output
            elif isinstance(raw_output, dict) and 'features' in raw_output:
                features = raw_output['features']
                if not isinstance(features, list): features = [features] # Ensure 'features' key maps to a list
            else:
                # If raw_output is a dict but not matching 'features', or other unexpected type
                raise RuntimeError(f"Model {self.terramind_model_name} forward() returned unexpected type or dict structure: {type(raw_output)}.")
        else:
            raise AttributeError(f"Model {self.terramind_model_name} lacks 'forward_features' or 'forward' method.")

        if features is None: # Should not happen if one of the branches above executed correctly
            raise RuntimeError(f"Failed to obtain features from model {self.terramind_model_name}")

        if isinstance(features, torch.Tensor): # Ensure list output, as UPerNet expects a list of features
            features = [features]
        elif not (isinstance(features, list) and all(isinstance(f, torch.Tensor) for f in features)):
            # If it's a list, but not of tensors, or not a list at all (and not a single tensor)
            raise RuntimeError(f"Features obtained from model are not a tensor or a list of tensors. Type: {type(features)}, Content: {features}")
        

        # Extract features at specified indices with bounds checking
        if isinstance(features, list) and self.output_layers is not None:
            features = [features[idx] for idx in self.output_layers if 0 <= idx < len(features)]

        # Convert Vision Transformer features to convolutional format expected by UPerNet
        # TerraMind models output features in [batch, seq_len, embed_dim] format
        # We need to reshape them to [batch, embed_dim, height, width] format
        # print(f"  DEBUG: Raw features shapes before reshape: {[f.shape for f in features]}")
        
        reshaped_features = []
        patch_size = 16  # TerraMind models typically use 16x16 patches
        for i, feat in enumerate(features):
            if len(feat.shape) == 3:  # [batch, seq_len, embed_dim]
                batch_size, seq_len, embed_dim = feat.shape
                
                # Calculate spatial dimensions
                # For 224x224 input with 16x16 patches: seq_len = 196 = 14*14
                spatial_size = int(seq_len ** 0.5)
                
                # Reshape from [batch, seq_len, embed_dim] to [batch, embed_dim, height, width]
                feat_reshaped = feat.transpose(1, 2).view(
                    batch_size, embed_dim, spatial_size, spatial_size
                ).contiguous()
                
                reshaped_features.append(feat_reshaped)
                # print(f"  DEBUG: Reshaped feature {i}: {feat.shape} -> {feat_reshaped.shape}")
            else:
                # Feature is already in the correct format
                reshaped_features.append(feat)
                # print(f"  DEBUG: Feature {i} already in correct format: {feat.shape}")

        # print(f"  TerraMindAllEncoder: Returning {len(reshaped_features)} feature maps.")
        return reshaped_features

    def load_encoder_weights(self, logger: Logger) -> None:
        """
        Load encoder weights. For this class, `pretrained=True` (or the value of
        `self.terramind_pretrained`) in `BACKBONE_REGISTRY.build` is expected to handle
        weight loading. This method serves to comply with the Encoder base class interface
        and log information.
        """
        if self.terramind_pretrained:
            logger.info(
                f"Pretrained weights for '{self.model_name}' (underlying: '{self.terramind_model_name}') "
                f"are expected to be loaded by terratorch via `pretrained=True` during model build."
            )
            
            # Basic check to see if parameters exist
            num_params = sum(p.numel() for p in self.model.parameters())
            # More specific check for trainable parameters
            num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            if num_params > 0:
                logger.info(
                    f"Model '{self.terramind_model_name}' has {num_params} total parameters "
                    f"({num_trainable_params} trainable), suggesting weights are present."
                )
            else:
                logger.warning(
                    f"Model '{self.terramind_model_name}' has 0 parameters. "
                    "Weights might not be loaded, or the model architecture is empty."
                )
        else:
            logger.info(
                f"Model '{self.terramind_model_name}' is configured with `pretrained=False`. "
                "No pretrained weights will be loaded by terratorch for this instance. "
                "If you have a local weights file, you might need to load it manually here "
                "using `torch.load()` and `self.model.load_state_dict()`."
            )
        
        # The base class's `download_model()` is called in `super().__init__()`.
        # If `self.download_url` was provided and `self.encoder_weights` points to a specific file,
        # that mechanism would attempt to download/load weights.
        # However, for terratorch with `pretrained=True`, this is usually not needed.
        pass

# Example of how this encoder might be instantiated in a config file (e.g., YAML):
# (This is conceptual and depends on your configuration system)