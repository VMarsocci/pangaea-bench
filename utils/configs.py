import yaml


def load_specific_config(args, key, train_config={}):
    if args.get(key):
        with open(args[key], "r") as file:
            return yaml.safe_load(file)
    elif train_config.get(key):
        with open(train_config[key], "r") as file:
            return yaml.safe_load(file)
    else:
        raise ValueError(f"No configuration found for {key}")


def load_config(args):
    cfg_path = args["run_config"]
    with open(cfg_path, "r") as file:
        train_config = yaml.safe_load(file)

    encoder_config = load_specific_config(args, "encoder_config", train_config=train_config)
    dataset_config = load_specific_config(args, "dataset_config", train_config=train_config) 
    task_config = load_specific_config(args, "task_config", train_config=train_config)

    # Add task_config parameters from dataset
    if dataset_config.get("num_classes"):
        task_config["head_args"]["num_classes"] = dataset_config["num_classes"]

    # Validate config
    if dataset_config.get("img_size") and encoder_config["encoder_model_args"].get("img_size"):
        if dataset_config["img_size"] != encoder_config["encoder_model_args"]["img_size"]:
            print(f"Warning: dataset img_size {dataset_config['img_size']} and encoder img_size {encoder_config['encoder_model_args']['img_size']} do not match. {encoder_config['encoder_model_args']['img_size']} is used.") 
        task_config["img_size"] = encoder_config["encoder_model_args"]["img_size"]
    
    if not dataset_config["multi_temporal"] and task_config["head_args"]["num_frames"] > 1:
        raise ValueError("task head num_frame > 1 is only supported for multi_temporal datasets.")

    return train_config, encoder_config, dataset_config, task_config