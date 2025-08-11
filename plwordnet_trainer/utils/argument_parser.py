from argparse import ArgumentParser

INPUT_FILE_REQUIRED = "input_file_required"
INPUT_FILE_NOT_REQUIRED = "input_file_not_required"
INPUT_DIR_REQUIRED = "input_dir_required"
INPUT_DIR_NOT_REQUIRED = "input_dir_not_required"
OUTPUT_FILE_REQUIRED = "output_file_required"
OUTPUT_FILE_NOT_REQUIRED = "output_file_not_required"
OUTPUT_DIR_REQUIRED = "output_dir_required"
OUTPUT_DIR_NOT_REQUIRED = "output_dir_not_required"
WORK_DIR_REQUIRED = "work_dir_required"
WORK_DIR_NOT_REQUIRED = "work_dir_not_required"
BASE_MODEL_REQUIRED = "base_model_required"
BASE_MODEL_NOT_REQUIRED = "base_model_not_required"
REWARD_MODEL_REQUIRED = "reward_model_required"
REWARD_MODEL_NOT_REQUIRED = "reward_model_not_required"
TRAIN_INPUT_FILE_REQUIRED = "train_input_file_required"
TRAIN_INPUT_FILE_NOT_REQUIRED = "train_input_file_not_required"
TEST_INPUT_FILE_REQUIRED = "test_input_file_required"
TEST_INPUT_FILE_NOT_REQUIRED = "test_input_file_not_required"
VALID_INPUT_FILE_REQUIRED = "valid_input_file_required"
VALID_INPUT_FILE_NOT_REQUIRED = "valid_input_file_not_required"
WANDB_BOOLEAN_FULL = "wandb_boolean_full"


class _APM:
    """
    Argument Parser Mapper
    """

    class APMD:
        """
        Argument parsr mapper definitions
        """

        PMAP_SHORT = "short"
        PMAP_LONG = "long"
        PMAP_DESCR = "description"
        PMAP_DEST = "dest"
        PMAP_REQ = "required"
        PMAP_TYPE = "type"

        INPUT_FILE = {
            PMAP_SHORT: "-i",
            PMAP_LONG: "--input-file",
            PMAP_DEST: "input_file",
            PMAP_DESCR: "Path to input file",
            PMAP_TYPE: "str",
        }
        INPUT_DIR = {
            PMAP_SHORT: "-I",
            PMAP_LONG: "--input-dir",
            PMAP_DEST: "input_dir",
            PMAP_DESCR: "Path to input directory",
            PMAP_TYPE: "str",
        }
        OUTPUT_FILE = {
            PMAP_SHORT: "-o",
            PMAP_LONG: "--output-file",
            PMAP_DEST: "output_file",
            PMAP_DESCR: "Output file",
            PMAP_TYPE: "str",
        }
        OUTPUT_DIR = {
            PMAP_SHORT: "-O",
            PMAP_LONG: "--output-dir",
            PMAP_DEST: "output_dir",
            PMAP_DESCR: "Output directory",
            PMAP_TYPE: "str",
        }
        WORK_DIR = {
            PMAP_SHORT: "-W",
            PMAP_LONG: "--work-dir",
            PMAP_DEST: "work_dir",
            PMAP_DESCR: "Path to temporary work directory",
            PMAP_TYPE: "str",
        }
        TRAIN_INPUT_FILE = {
            PMAP_SHORT: "-T",
            PMAP_LONG: "--train-file",
            PMAP_DEST: "train_file",
            PMAP_DESCR: "Path to train input file",
            PMAP_TYPE: "str",
        }
        TEST_INPUT_FILE = {
            PMAP_SHORT: "-TT",
            PMAP_LONG: "--test-file",
            PMAP_DEST: "test_file",
            PMAP_DESCR: "Path to test input file",
            PMAP_TYPE: "str",
        }
        VALID_INPUT_FILE = {
            PMAP_SHORT: "-V",
            PMAP_LONG: "--valid-file",
            PMAP_DEST: "valid_file",
            PMAP_DESCR: "Path to train input file",
            PMAP_TYPE: "str",
        }
        BASE_MODEL = {
            PMAP_SHORT: "-m",
            PMAP_LONG: "--base-model",
            PMAP_DEST: "base_model",
            PMAP_DESCR: "Name or path to base model to be used.",
            PMAP_TYPE: "str",
        }
        REWARD_MODEL = {
            PMAP_SHORT: "-rlrm",
            PMAP_LONG: "--reward-model",
            PMAP_DEST: "reward_model",
            PMAP_DESCR: "Name or path to reward model to be used in case of reinforcement learning.",
            PMAP_TYPE: "str",
        }
        WANDB_ONOFF = {
            PMAP_SHORT: "-WB",
            PMAP_LONG: "--use-wandb-full",
            PMAP_DEST: "wandb_full",
            PMAP_DESCR: "If enabled then full W&B integration will be enabled. Full "
            "logging, dataset as artifact storing and models storing",
            PMAP_TYPE: "bool",
        }

    INPUT_FILE_REQUIRED = APMD.INPUT_FILE.copy()
    INPUT_FILE_REQUIRED[APMD.PMAP_REQ] = True
    INPUT_FILE_NOT_REQUIRED = APMD.INPUT_FILE.copy()
    INPUT_FILE_NOT_REQUIRED[APMD.PMAP_REQ] = False

    INPUT_DIR_REQUIRED = APMD.INPUT_DIR.copy()
    INPUT_DIR_REQUIRED[APMD.PMAP_REQ] = True
    INPUT_DIR_NOT_REQUIRED = APMD.INPUT_FILE.copy()
    INPUT_DIR_NOT_REQUIRED[APMD.PMAP_REQ] = False

    OUTPUT_FILE_REQUIRED = APMD.OUTPUT_FILE.copy()
    OUTPUT_FILE_REQUIRED[APMD.PMAP_REQ] = True
    OUTPUT_FILE_NOT_REQUIRED = APMD.OUTPUT_FILE.copy()
    OUTPUT_FILE_NOT_REQUIRED[APMD.PMAP_REQ] = False

    OUTPUT_DIR_REQUIRED = APMD.OUTPUT_DIR.copy()
    OUTPUT_DIR_REQUIRED[APMD.PMAP_REQ] = True
    OUTPUT_DIR_NOT_REQUIRED = APMD.OUTPUT_DIR.copy()
    OUTPUT_DIR_NOT_REQUIRED[APMD.PMAP_REQ] = False

    WORK_DIR_REQUIRED = APMD.WORK_DIR.copy()
    WORK_DIR_REQUIRED[APMD.PMAP_REQ] = True
    WORK_DIR_NOT_REQUIRED = APMD.WORK_DIR.copy()
    WORK_DIR_NOT_REQUIRED[APMD.PMAP_REQ] = False

    BASE_MODEL_REQUIRED = APMD.BASE_MODEL.copy()
    BASE_MODEL_REQUIRED[APMD.PMAP_REQ] = True
    BASE_MODEL_NOT_REQUIRED = APMD.BASE_MODEL.copy()
    BASE_MODEL_NOT_REQUIRED[APMD.PMAP_REQ] = False

    REWARD_MODEL_REQUIRED = APMD.REWARD_MODEL.copy()
    REWARD_MODEL_REQUIRED[APMD.PMAP_REQ] = True
    REWARD_MODEL_NOT_REQUIRED = APMD.REWARD_MODEL.copy()
    REWARD_MODEL_NOT_REQUIRED[APMD.PMAP_REQ] = False

    TRAIN_FILE_REQUIRED = APMD.TRAIN_INPUT_FILE.copy()
    TRAIN_FILE_REQUIRED[APMD.PMAP_REQ] = True
    TRAIN_FILE_NOT_REQUIRED = APMD.TRAIN_INPUT_FILE.copy()
    TRAIN_FILE_NOT_REQUIRED[APMD.PMAP_REQ] = False

    TEST_FILE_REQUIRED = APMD.TEST_INPUT_FILE.copy()
    TEST_FILE_REQUIRED[APMD.PMAP_REQ] = True
    TEST_FILE_NOT_REQUIRED = APMD.TEST_INPUT_FILE.copy()
    TEST_FILE_NOT_REQUIRED[APMD.PMAP_REQ] = False

    VALID_FILE_REQUIRED = APMD.VALID_INPUT_FILE.copy()
    VALID_FILE_REQUIRED[APMD.PMAP_REQ] = True
    VALID_FILE_NOT_REQUIRED = APMD.VALID_INPUT_FILE.copy()
    VALID_FILE_NOT_REQUIRED[APMD.PMAP_REQ] = False

    WANDB_BOOL_ONOFF = APMD.WANDB_ONOFF.copy()


# There are mappings param name to argument parser item
PARSER_FIELDS_MAP = {
    INPUT_FILE_REQUIRED: _APM.INPUT_FILE_REQUIRED,
    INPUT_FILE_NOT_REQUIRED: _APM.INPUT_FILE_NOT_REQUIRED,
    INPUT_DIR_REQUIRED: _APM.INPUT_DIR_REQUIRED,
    INPUT_DIR_NOT_REQUIRED: _APM.INPUT_DIR_NOT_REQUIRED,
    OUTPUT_FILE_REQUIRED: _APM.OUTPUT_FILE_REQUIRED,
    OUTPUT_FILE_NOT_REQUIRED: _APM.OUTPUT_FILE_NOT_REQUIRED,
    OUTPUT_DIR_REQUIRED: _APM.OUTPUT_DIR_REQUIRED,
    OUTPUT_DIR_NOT_REQUIRED: _APM.OUTPUT_DIR_NOT_REQUIRED,
    WORK_DIR_REQUIRED: _APM.WORK_DIR_REQUIRED,
    WORK_DIR_NOT_REQUIRED: _APM.WORK_DIR_NOT_REQUIRED,
    BASE_MODEL_REQUIRED: _APM.BASE_MODEL_REQUIRED,
    BASE_MODEL_NOT_REQUIRED: _APM.BASE_MODEL_NOT_REQUIRED,
    REWARD_MODEL_REQUIRED: _APM.REWARD_MODEL_REQUIRED,
    REWARD_MODEL_NOT_REQUIRED: _APM.REWARD_MODEL_NOT_REQUIRED,
    TRAIN_INPUT_FILE_REQUIRED: _APM.TRAIN_FILE_REQUIRED,
    TRAIN_INPUT_FILE_NOT_REQUIRED: _APM.TRAIN_FILE_NOT_REQUIRED,
    TEST_INPUT_FILE_REQUIRED: _APM.TEST_FILE_REQUIRED,
    TEST_INPUT_FILE_NOT_REQUIRED: _APM.TEST_FILE_NOT_REQUIRED,
    VALID_INPUT_FILE_REQUIRED: _APM.VALID_FILE_REQUIRED,
    VALID_INPUT_FILE_NOT_REQUIRED: _APM.VALID_FILE_NOT_REQUIRED,
    WANDB_BOOLEAN_FULL: _APM.WANDB_BOOL_ONOFF,
}


def base_argument_parser(desc=""):
    p = ArgumentParser(description=desc)
    return p


def prepare_parser_for_fields(
    fields_list: list, description: str = None
) -> ArgumentParser:
    """
    It's a general utility function which may be used to prepare ArgumentParser
    with predefined options.

    :param fields_list: List of fields to construct the parser
    :param description: Application/parser description
    :return: Object of ArgumentParser
    """
    p = base_argument_parser(description if description is not None else "")

    for item in fields_list:
        if item not in PARSER_FIELDS_MAP:
            raise Exception(f"Cannot find argument parser mapping for item: {item} ")

        p_item = PARSER_FIELDS_MAP[item]

        if p_item[_APM.APMD.PMAP_TYPE] in ["bool"]:
            p.add_argument(
                p_item[_APM.APMD.PMAP_SHORT],
                p_item[_APM.APMD.PMAP_LONG],
                help=p_item[_APM.APMD.PMAP_DESCR],
                dest=p_item[_APM.APMD.PMAP_DEST],
                action="store_true",
            )
        else:
            p.add_argument(
                p_item[_APM.APMD.PMAP_SHORT],
                p_item[_APM.APMD.PMAP_LONG],
                help=p_item[_APM.APMD.PMAP_DESCR],
                dest=p_item[_APM.APMD.PMAP_DEST],
                required=bool(
                    p_item[_APM.APMD.PMAP_REQ],
                ),
            )
    return p
