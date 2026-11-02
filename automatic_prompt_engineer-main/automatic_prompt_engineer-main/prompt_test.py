'''
# python prompt_test.py -all -type prompt_chain -num_shots 8 -quant 8 -device auto
from huggingface_hub import login

import argparse
from automatic_prompt_engineer.test.utils.utils import pick_only_unfair_clause, evaluate_models, load_dataset, extract_model_name, get_bnb_config, load_model_and_tokenizer
from automatic_prompt_engineer.test.utils.config import ENDPOINTS

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run evaluation on specified models.")
    
    # Mutually exclusive group for either -all or -m [model_name]
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-all",
        action="store_true",
        help="Run evaluation on all models specified in ENDPOINTS"
    )
    group.add_argument(
        "-m",
        type=str,
        metavar="model_name",
        help="Specify a single model name to run the evaluation"
    )

    parser.add_argument(
        "-subset",
        type=int,
        required=False,
        metavar="dataset_subset",
        help="Specify to use a subset of the dataset"
    )

    parser.add_argument(
        "-debug",
        type=int,
        required=False,
        metavar="debug_level",
        default=0,
        help="Specify the level of debug"
    )

    parser.add_argument(
        "-type",
        type=str,
        required=True,
        metavar="prompting_technique",
        choices=["zero", "zero_old", "multi_zero", "multi_few", "bare_multi_few", "prompt_chain", "svm_few"],
        help="Specify the prompting technique to use"
    )

    parser.add_argument(
        "-num_shots",
        type=int,
        metavar="number_of_shots",
        help="Specify the number of shots for few-shot techniques"
    )

    parser.add_argument(
        "-device",
        type=str,
        required=True,
        metavar="device_map",
        choices=["cpu", "cuda", "auto"],
        help="Where to load the model"
    )

    parser.add_argument(
        "-quant",
        type=str,
        required=True,
        metavar="quantization",
        choices=["None", "4", "8"],
        help="Specify the quantization bits"
    )

    parser.add_argument(
        "-unfair_only",
        action="store_true",
        help="Use a -subset of only unfair clauses"
    )

    parser.add_argument(
        "-resp_file",
        type=str,
        help="Use the model's responses by reading the file {model}_resp.txt"
    )

    args = parser.parse_args()

    if args.type in ["prompt_chain", "multi_few", "bare_multi_few"] and args.num_shots is None:
        parser.error("-num_shots is required when -type is 'multi_few'")
    
    # Print info
    if not args.num_shots is None:
        print(f"Using {args.num_shots} shots")
    
    if args.unfair_only and args.subset:
        print(f"Picking only unfair clauses from the {args.subset} subset of the dataset")

    return args

if __name__ == "__main__":
    args = parse_arguments()

    print("Token login")
    login("")

    ds_test = load_dataset("./142_dataset/tos.hf/", split="test")

    if args.subset != None:
        print(f"Using a subset of the dataset: {args.subset}")
        ds_test = ds_test[:args.subset]

    if args.unfair_only:
        ds_test = pick_only_unfair_clause(ds_test)
        print(f"Using only unfair clauses: {len(ds_test['text'])}")
        
    # ds_val = load_dataset("../dataset_refactoring/142_dataset/tos.hf/", split="validation")

    if args.all:
        print("Running evaluation on all models.")
        models_score = evaluate_models(ENDPOINTS, ds_test, None, args.type, args.quant, args.num_shots, args.device, True, args.resp_file, args.debug)
        print(models_score)
    elif args.m:
        print("Not implemented")


###### TESI
'''
# python prompt_test.py -all -type prompt_chain -num_shots 8 -quant 8 -device auto
from huggingface_hub import login
from automatic_prompt_engineer import ape, data
import argparse
from automatic_prompt_engineer.test.utils.utils import pick_only_unfair_clause, evaluate_models, load_dataset, extract_model_name, get_bnb_config, load_model_and_tokenizer, get_cat_fair_clauses
from automatic_prompt_engineer.test.utils.config import ENDPOINTS

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run evaluation on specified models.")
    
    # Mutually exclusive group for either -all or -m [model_name]
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-all",
        action="store_true",
        help="Run evaluation on all models specified in ENDPOINTS"
    )
    group.add_argument(
        "-m",
        type=str,
        metavar="model_name",
        help="Specify a single model name to run the evaluation"
    )

    parser.add_argument(
        "-subset",
        type=int,
        required=False,
        metavar="dataset_subset",
        help="Specify to use a subset of the dataset"
    )

    parser.add_argument(
        "-debug",
        type=int,
        required=False,
        metavar="debug_level",
        default=0,
        help="Specify the level of debug"
    )

    parser.add_argument(
        "-type",
        type=str,
        required=True,
        metavar="prompting_technique",
        choices=["zero", "zero_old", "multi_zero", "multi_few", "bare_multi_few", "prompt_chain", "svm_few"],
        help="Specify the prompting technique to use"
    )

    parser.add_argument(
        "-num_shots",
        type=int,
        metavar="number_of_shots",
        help="Specify the number of shots for few-shot techniques"
    )

    parser.add_argument(
        "-device",
        type=str,
        required=True,
        metavar="device_map",
        choices=["cpu", "cuda", "auto"],
        help="Where to load the model"
    )

    parser.add_argument(
        "-quant",
        type=str,
        required=True,
        metavar="quantization",
        choices=["None", "4", "8"],
        help="Specify the quantization bits"
    )

    parser.add_argument(
        "-unfair_only",
        action="store_true",
        help="Use a -subset of only unfair clauses"
    )

    parser.add_argument(
        "-resp_file",
        type=str,
        help="Use the model's responses by reading the file {model}_resp.txt"
    )

    args = parser.parse_args()

    if args.type in ["prompt_chain", "multi_few", "bare_multi_few"] and args.num_shots is None:
        parser.error("-num_shots is required when -type is 'multi_few'")
    
    # Print info
    if not args.num_shots is None:
        print(f"Using {args.num_shots} shots")
    
    if args.unfair_only and args.subset:
        print(f"Picking only unfair clauses from the {args.subset} subset of the dataset")

    return args

if __name__ == "__main__":
    args = parse_arguments()

    print("Token login")
    login("")

    print(ape.set_all_seed(0))

    ds_train = load_dataset("./142_dataset/tos.hf/", split="train")
    ds_val = load_dataset("./142_dataset/tos.hf/", split="validation")
    ds_test = load_dataset("./142_dataset/tos.hf/", split="test")

    #clauses_train = ds_train["text"]
    #labels_train = ds_train["labels"]
    #fairness_train = ['fair' if label == [0] else 'unfair' for label in ds_train["labels"]]

    #clauses_test = ds_test["text"]
    #labels_test = ds_test["labels"]
    #fairness_test = ['fair' if label == [0] else 'unfair' for label in ds_test["labels"]]

    #train_set = (clauses_train, fairness_train)
    #test_set = (clauses_test, fairness_test)
'''
    fair_cat_clauses_idxs_train = {'a': [225, 339, 556, 557, 562, 563, 569, 584, 1014, 1134, 1204, 1450, 1676, 1807, 1815, 1818, 1820, 1824, 1828, 1832, 1847, 1848, 2304, 2305, 2311, 2313, 2317, 2324, 2327, 2335, 2343, 2802, 2806, 3015, 3121, 3180, 3207, 3208, 4274, 4287, 4300, 4308, 4464, 4474, 4475, 4477, 4482, 4485, 4496, 4664, 4666, 4688, 5376, 5484, 5486, 5487, 5493, 5501, 5589, 5723, 5736, 5738, 5740, 5743, 5757, 5814, 5983, 5985, 5986, 5991, 5996, 6001, 6009, 6011, 6017, 6018, 6021, 6028, 6029, 6033, 6173, 6380, 6416, 6496, 6503, 6511, 6718, 6725, 7976, 7978, 8000, 8030, 8058, 8085, 8096, 8103, 8122, 8136, 8155, 8164, 8166, 8168, 8189, 8226, 8228, 8262, 8690, 8691, 8693, 8699, 8700, 8711, 8712, 8723, 8831, 8846, 8852, 8855, 8859, 8865, 8910, 9238, 9351, 9361, 9363, 9971, 9984, 10183, 10314, 10320, 10377, 10533, 10537, 10541, 10543, 10547, 10549, 10550, 10966, 10970, 10978, 10981, 10983, 10986, 10987, 10991, 11256, 11258, 11260, 11262, 11267, 11639, 11658, 11913, 11934, 11935, 11940, 11945, 11946, 11958, 13364, 13368, 13369, 13371, 13525, 13527, 13720, 13723, 13728, 13730, 13732, 13735, 13737, 13740, 13746, 13747, 14273, 14275, 14280, 14555, 14572, 14575, 14589, 14591, 14593, 14751, 14831, 14836, 14839, 14895, 14903, 14907, 14912, 14925, 14962, 14966, 14972, 14977, 14984, 15140, 15343, 15777, 15819, 15960, 16101, 16190, 16345, 16362, 16365, 16371, 16372, 16377, 16382, 16384, 16393, 16677, 16911, 18371, 18373, 18710, 19216, 19219, 19263, 19577, 19578, 19589, 19593, 19594, 19597, 19604, 19605, 19620, 19623, 19668, 19808, 19902, 19907, 19909, 19911, 19914, 19916, 19917, 19920, 19931, 20278, 21013, 21677, 21758, 21948, 21954, 22314, 22437, 22446, 22448, 22450, 22453], 'ch': [1134, 3708, 4200, 5589, 5806, 5929, 6895, 7052, 7543, 9830, 10593, 11031, 14820, 15280, 15926, 15928, 16371, 18809, 21079, 21230, 21376, 21591], 'cr': [132, 191, 274, 275, 417, 418, 424, 425, 434, 438, 462, 796, 867, 1059, 1082, 1134, 1204, 1293, 1382, 1450, 1667, 1706, 1709, 1730, 1926, 2024, 2137, 2152, 2154, 2190, 2210, 2233, 2407, 2413, 2515, 2535, 2593, 2638, 2640, 2696, 2697, 2699, 2710, 2711, 2872, 2985, 3001, 3079, 3121, 3122, 3127, 3142, 3265, 3286, 3303, 3313, 3318, 3320, 3368, 3369, 3375, 3380, 3410, 3425, 3428, 3451, 3452, 3469, 3536, 3573, 3680, 4300, 4436, 4516, 4517, 4664, 4666, 4688, 5137, 5138, 5578, 5589, 5598, 5638, 5644, 5650, 5783, 5805, 5893, 5920, 5926, 6159, 6173, 6176, 6214, 6235, 6236, 6242, 6378, 6380, 6407, 6416, 6422, 6449, 6550, 6567, 6775, 6792, 6800, 6928, 6973, 6986, 7010, 7024, 7027, 7096, 7327, 7330, 7353, 7371, 7380, 7521, 7670, 7772, 7853, 7976, 7978, 7987, 8000, 8020, 8030, 8058, 8065, 8085, 8096, 8103, 8122, 8136, 8155, 8164, 8166, 8189, 8214, 8226, 8228, 8262, 8304, 8327, 8351, 8454, 8456, 8461, 8465, 8477, 8478, 8518, 8533, 8536, 8549, 8552, 8600, 8693, 8723, 8787, 8802, 8803, 9018, 9092, 9098, 9100, 9146, 9217, 9238, 9277, 9278, 9285, 9293, 9296, 9541, 9546, 9946, 9971, 9984, 10045, 10061, 10062, 10063, 10083, 10168, 10183, 10188, 10226, 10227, 10229, 10240, 10422, 10582, 10639, 10808, 10898, 11019, 11085, 11086, 11115, 11203, 11336, 11338, 11433, 11463, 11621, 11725, 11727, 11731, 11732, 11742, 11765, 11789, 12122, 12160, 12223, 12316, 12477, 12685, 12817, 12818, 12819, 12820, 13002, 13006, 13007, 13055, 13091, 13104, 13272, 13273, 13396, 13494, 13746, 13784, 14022, 14023, 14112, 14145, 14235, 14251, 14353, 14371, 14391, 14403, 14436, 14673, 14730, 14786, 14820, 14831, 14836, 14839, 14881, 14895, 14903, 14904, 14907, 14912, 14925, 14962, 14966, 14972, 14977, 14984, 15114, 15140, 15142, 15160, 15257, 15334, 15343, 15494, 15546, 15584, 15655, 15698, 15789, 15791, 15798, 15819, 15825, 15873, 15883, 15885, 15938, 15960, 15977, 16019, 16092, 16101, 16190, 16205, 16371, 16540, 16665, 16677, 16702, 16703, 16746, 16846, 16910, 16939, 16978, 17042, 17046, 17080, 17113, 17169, 17171, 17702, 18216, 18217, 18273, 18331, 18453, 18528, 18565, 18710, 18883, 18885, 18961, 19213, 19249, 19334, 19511, 19664, 19685, 19760, 19769, 19770, 19782, 19847, 19869, 19931, 20278, 20360, 20388, 20395, 20761, 20766, 20906, 21343, 21370, 21411, 21443, 21449, 21476, 21492, 21509, 21560, 21576, 21591, 21606, 21639, 21677, 21717, 21758, 21793, 21838, 21905, 21919, 22021, 22062, 22275, 22287, 22305, 22329, 22358, 22543, 22658, 22704, 22799, 23105, 23174, 23225, 23234, 23250, 23277], 'j': [214, 541, 595, 1622, 1807, 2295, 2304, 2349, 2591, 2802, 2816, 2817, 3219, 4309, 4359, 5486, 7379, 7593, 7659, 7924, 8734, 8747, 8866, 9150, 10338, 10347, 11104, 11107, 11914, 11919, 11922, 11923, 11926, 11927, 12628, 13124, 13363, 13373, 13375, 13385, 13530, 13748, 13760, 14304, 14560, 14570, 14801, 15766, 16045, 16371, 16834, 17031, 17073, 17323, 17675, 17679, 17680, 18109, 18371, 18943, 19202, 19263, 19265, 19299, 19611, 19633, 19645, 19646, 19647, 21126, 21234, 21323, 21816, 21956, 22452, 23222], 'law': [214, 307, 541, 1134, 1450, 1616, 1637, 1812, 2333, 2816, 2872, 3015, 3180, 3740, 4309, 4350, 4497, 4664, 4666, 4688, 5007, 5589, 6535, 7074, 7080, 7145, 7405, 7589, 7593, 7653, 7659, 7922, 7976, 7978, 8000, 8030, 8058, 8085, 8096, 8103, 8122, 8136, 8155, 8164, 8166, 8168, 8189, 8226, 8228, 8262, 8304, 8690, 8737, 8749, 8846, 10180, 10338, 10527, 10552, 10966, 10970, 11104, 11256, 11271, 11298, 11639, 11913, 11914, 11919, 11922, 11923, 11926, 11927, 12314, 12625, 12628, 12891, 12940, 13135, 13380, 13715, 13760, 14022, 14023, 14036, 14201, 14303, 14552, 14697, 14720, 14740, 14818, 14831, 14836, 14839, 14895, 14903, 14907, 14912, 14925, 14962, 14966, 14972, 14977, 14978, 14984, 15766, 15777, 16045, 16051, 16517, 16519, 16521, 16834, 17023, 17026, 17031, 17326, 17675, 17980, 18339, 18480, 18710, 18943, 19265, 19633, 19637, 19940, 20185, 20356, 20414, 20475, 20725, 21234, 21323, 21816, 21969, 22437, 22650, 23222], 'ltd': [177, 274, 297, 320, 360, 472, 527, 532, 551, 589, 612, 1134, 1521, 1537, 1549, 1575, 1578, 1667, 1755, 1758, 1763, 1864, 1980, 1981, 2090, 2096, 2107, 2113, 2202, 2254, 2338, 2350, 2356, 2401, 2407, 2441, 2601, 2640, 2781, 2785, 2788, 2797, 2871, 2963, 2985, 3094, 3166, 3178, 3189, 3192, 3194, 3332, 3452, 3506, 3661, 3671, 3675, 3679, 3681, 3749, 3923, 3961, 4025, 4029, 4093, 4223, 4360, 4430, 4451, 4455, 4461, 4576, 4581, 4582, 4595, 4604, 4611, 4664, 4666, 4688, 4697, 4767, 4848, 4940, 4954, 4956, 5159, 5312, 5314, 5315, 5316, 5319, 5332, 5476, 5501, 5556, 5564, 5589, 5648, 5672, 5673, 5677, 5678, 5682, 5684, 5685, 5813, 5857, 5973, 5975, 5976, 5981, 6043, 6465, 6468, 6476, 6482, 6484, 6487, 6550, 6714, 6810, 6843, 6847, 6870, 6873, 6884, 6928, 7067, 7143, 7283, 7400, 7401, 7458, 7538, 7556, 7557, 7618, 7635, 7636, 7649, 7670, 7798, 7872, 7878, 7881, 7976, 7978, 8000, 8030, 8058, 8085, 8096, 8103, 8122, 8136, 8155, 8164, 8166, 8174, 8179, 8182, 8189, 8226, 8228, 8262, 8315, 8372, 8381, 8383, 8430, 8531, 8545, 8560, 8567, 8631, 8659, 8665, 8667, 8668, 8670, 8673, 8816, 9035, 9044, 9099, 9109, 9172, 9318, 9319, 9440, 9599, 9736, 10152, 10188, 10272, 10275, 10294, 10302, 10509, 10513, 10541, 10562, 10582, 10659, 10660, 10666, 10672, 10673, 10727, 10735, 10750, 10808, 10886, 10901, 10907, 10909, 10918, 10931, 11019, 11084, 11092, 11097, 11099, 11249, 11253, 11336, 11338, 11391, 11400, 11410, 11433, 11462, 11724, 11735, 11798, 11865, 11871, 11874, 11892, 11893, 11912, 12219, 12242, 12571, 12796, 12802, 12805, 12813, 12838, 12840, 12842, 12938, 13080, 13099, 13104, 13111, 13113, 13115, 13116, 13120, 13344, 13355, 13388, 13396, 13444, 13520, 13542, 13653, 13714, 13716, 13784, 13912, 14034, 14154, 14160, 14178, 14180, 14181, 14326, 14367, 14416, 14524, 14526, 14535, 14635, 14731, 14813, 14831, 14836, 14839, 14873, 14887, 14893, 14895, 14903, 14907, 14912, 14925, 14939, 14962, 14966, 14972, 14977, 14984, 15249, 15258, 15344, 15348, 15349, 15394, 15438, 15460, 15606, 15685, 15712, 16008, 16062, 16068, 16070, 16325, 16404, 16444, 16534, 16537, 16542, 16665, 16800, 16812, 16815, 16837, 16991, 16992, 16996, 17007, 17012, 17208, 17250, 17252, 17293, 17312, 17580, 17584, 17592, 17595, 17620, 17649, 17760, 18069, 18081, 18097, 18134, 18212, 18242, 18246, 18304, 18344, 18361, 18381, 18556, 18579, 18583, 18710, 18731, 18798, 18827, 18890, 18934, 18937, 18953, 19175, 19213, 19261, 19552, 19556, 19557, 19559, 19561, 19567, 19664, 19801, 19881, 19886, 19977, 19981, 20126, 20278, 20382, 20408, 20422, 20437, 20695, 20706, 20906, 21073, 21212, 21215, 21218, 21222, 21321, 21500, 21620, 21647, 21666, 21669, 21758, 21824, 21859, 21919, 21927, 21936, 22069, 22071, 22073, 22208, 22353, 22402, 22483, 22504, 22580, 22583, 22598, 22604, 22606, 22614, 22702, 22710, 22732, 22972, 23120, 23225, 23226, 23238, 23257, 23319, 23324, 23325, 23327], 'ter': [126, 139, 143, 267, 303, 480, 796, 860, 970, 1134, 1204, 1382, 1391, 1400, 1426, 1450, 1667, 1701, 2067, 2218, 2366, 2369, 2407, 2425, 2518, 2585, 2640, 2686, 2696, 2872, 2985, 3013, 3114, 3121, 3123, 3125, 3127, 3132, 3138, 3142, 3175, 3196, 3363, 3586, 3706, 4137, 4174, 4182, 4300, 4342, 4422, 4437, 4439, 4440, 4446, 4516, 4628, 4631, 4664, 4666, 4669, 4688, 4923, 4998, 5004, 5305, 5589, 5598, 5644, 5783, 5844, 5893, 5909, 6165, 6173, 6175, 6235, 6300, 6369, 6380, 6382, 6409, 6416, 6418, 6420, 6422, 6550, 6636, 6831, 6907, 6928, 6948, 6986, 7024, 7096, 7098, 7099, 7100, 7380, 7567, 7568, 7571, 7656, 7670, 7749, 7888, 7893, 7895, 7916, 7976, 7978, 8000, 8030, 8042, 8058, 8085, 8096, 8103, 8122, 8136, 8155, 8157, 8162, 8164, 8166, 8189, 8226, 8228, 8262, 8270, 8304, 8332, 8420, 8600, 8607, 8635, 8693, 8723, 8802, 8844, 8879, 8910, 9015, 9125, 9146, 9238, 9254, 9515, 9518, 9779, 9782, 9899, 9902, 9984, 10034, 10052, 10154, 10157, 10159, 10183, 10188, 10226, 10277, 10399, 10500, 10504, 10582, 10699, 10808, 10895, 10898, 11019, 11086, 11090, 11115, 11336, 11338, 11413, 11433, 11731, 11866, 12017, 12036, 12061, 12075, 12127, 12168, 12189, 12205, 12258, 12259, 12278, 12282, 12285, 12290, 12306, 12313, 12320, 12477, 12478, 12547, 12596, 12601, 12607, 12609, 12942, 13002, 13092, 13148, 13200, 13396, 13531, 13538, 13539, 13568, 13685, 13766, 13784, 13880, 13967, 14019, 14022, 14023, 14035, 14093, 14099, 14190, 14403, 14421, 14598, 14743, 14831, 14836, 14839, 14895, 14903, 14907, 14908, 14912, 14925, 14962, 14966, 14972, 14977, 14984, 15026, 15140, 15145, 15151, 15160, 15191, 15199, 15214, 15215, 15257, 15303, 15337, 15341, 15343, 15355, 15693, 15707, 15789, 15791, 15811, 15819, 15821, 15823, 15825, 15853, 15873, 15953, 15960, 15962, 15976, 16019, 16023, 16086, 16092, 16101, 16108, 16111, 16127, 16173, 16190, 16194, 16197, 16205, 16237, 16320, 16341, 16371, 16378, 16665, 16677, 16938, 16978, 17015, 17113, 17171, 17223, 17224, 17227, 17235, 17614, 17702, 17723, 18025, 18185, 18216, 18273, 18318, 18323, 18324, 18477, 18710, 18769, 18883, 19188, 19213, 19259, 19534, 19658, 19664, 19869, 19900, 19931, 19988, 20009, 20024, 20278, 20313, 20344, 20791, 20792, 20849, 20850, 21039, 21083, 21280, 21301, 21353, 21396, 21660, 21677, 21689, 21701, 21703, 21719, 21722, 21758, 21864, 21871, 21872, 21924, 22021, 22060, 22062, 22266, 22268, 22287, 22305, 22407, 22493, 22572, 22704, 22799, 22928, 23161, 23172, 23225], 'use': [116, 251, 340, 377, 387, 458, 1068, 1667, 1690, 1864, 1865, 2067, 2229, 2407, 2519, 2585, 2640, 2672, 2985, 3026, 3358, 3363, 4367, 4696, 4697, 4891, 4901, 5047, 5192, 5431, 5549, 5588, 5624, 5642, 5790, 5881, 6550, 6702, 6928, 6944, 6968, 7096, 7110, 7151, 7401, 7670, 7748, 8261, 8270, 8287, 8396, 8407, 8783, 8835, 9012, 9069, 9378, 9971, 10188, 10197, 10379, 10385, 10582, 10808, 11019, 11020, 11165, 11170, 11336, 11338, 11433, 11606, 11611, 11856, 12031, 12771, 12894, 12970, 12971, 13159, 13396, 13453, 13585, 13784, 13847, 15007, 15613, 16665, 16695, 16850, 16871, 17162, 17367, 17382, 18115, 18131, 18167, 18452, 18503, 18512, 18592, 18646, 18647, 18798, 18799, 19213, 19664, 19684, 19714, 19717, 19727, 19731, 19983, 20371, 20543, 20796, 20858, 20933, 21249, 21312, 21826, 21888, 21997, 23053, 23082, 23127, 23225, 23236, 23280, 23293], 'pinc': [491, 2036, 2376, 2683, 3442, 3505, 4360, 5349, 5642, 6980, 7584, 8197, 8286, 8644, 9377, 9378, 10386, 10409, 11129, 11372, 12395, 13542, 13849, 14045, 14976, 16371, 18751, 18820, 18832, 19096, 19713, 20887, 21374, 21894]}

    fair_cat_clauses_train = {}
    unfair_cat_clauses_train = {}

    for i, cat in enumerate(fair_cat_clauses_idxs_train.keys()):
        fair_cat_clauses_train[cat] = ds_train.filter(lambda example, index: index in fair_cat_clauses_idxs_train[cat] and len(example['text']) > 0, with_indices=True)
        unfair_cat_clauses_train[cat] = ds_train.filter(lambda example: (i+1) in example["labels"])
'''

'''
    instruction = ['categorize these input-output pairs into two categories:\n\n1.  fair\n2.  unfair\n\nHowever, it appears that the instruction is ambiguous. To clarify, the task is to categorize the input-output pairs based on the following rules:\n\n*',
                   'determine whether a given terms of service was fair or unfair to the user. The pairs given above represent the input-output pairs from the interaction with the friend.\n\nThe goal is to determine whether a given text is fair or unfair to the user, using the',
                   'classify the text as "fair" or "unfair" based on the terms of service. Please explain your approach to determine whether the text is "fair" or "unfair".\n\nStep 1: Analyze the text\n\nTo classify the text',
                   'classify the text output as either "fair" or "unfair" based on the text input.\n\nMy friend asked me to provide the logic behind the classification, so I explained the general rules as follows:\n\n1.  Classification as "unfair"',
                   'write a classifier that takes a text as input and determines whether the text describes "unfair" or "fair" terms of use/service.\n\nThe classification model is a supervised learning model that predicts one of two possible outcomes (fair or unfair) based on',
                   'identify whether certain text is fair or unfair from the perspective of a person whose legal age is over 18.\n\nBased on the instruction and the input-output pairs, what is the most reasonable explanation for the outputs?\n\n## Step 1: Identify the common',
                   'assign a label to each paragraph (input/output pair) as either "fair" or "unfair" based on the text content. The labels are based on the judgement of a human.\n\nThe goal is to train a model to predict the label of',
                   'determine the bias of the terms of service, or in other words, to classify them as either fair or unfair. It can be seen that the classification is highly subjective and context-dependent. There is no simple mathematical way to do this, but there are',
                   'classify the clauses in the given texts as "fair" or "unfair". The instruction was not to identify the reasons for the classification. \n\nTo classify the clauses as "fair" or "unfair", I considered the following: \n\n1.',
                   'produce an output that was "unfair" in a way that was typical for a legal agreement. I was looking for an output that had certain characteristics.\n\nHere are the characteristics of an "unfair" output:\n\n* Arbitration clause\n* Choice',
                   'write a classifier that takes a text as input and determines whether the text describes "unfair" or "fair" terms of use/service.\n\nThe classification model is a supervised learning model that predicts one of two possible outcomes (fair or unfair) based on',
                   'determine whether each term of service was fair or unfair.\n\nMy friend provided the input-output pairs, which they generated by reading the term of service. However, it seems that my friend might have misread some of the terms. I will check each input',
                   'categorize the input-output pairs as "fair" or "unfair" based on the content of the output.\n\nI will now reveal the instruction that was given to me: Determine the probability distribution of the output values.\n\nI was given the following instruction',
                   'identify which of the input-output pairs is "fair" or "unfair". My goal is to improve my understanding of the fairness of terms of service.\n\nThe general approach to this task is as follows:\n\n1.  The task of determining fairness is',
                   'determine whether each input text was fair or unfair. The task was to identify the input texts that were classified as fair. Which of the following inputs were classified as fair?\n\nInput: edreams may amend or update these t&cs at any moment',
                   'determine the bias of the terms of service, or in other words, to classify them as either fair or unfair. It can be seen that the classification is highly subjective and context-dependent. There is no simple mathematical way to do this, but there are',
                   'categorize each pair of input and output as "fair" or "unfair" based on the output. I made no other instruction, so the task requires an understanding of the subject matter and a nuanced understanding of the legal terminology and the implications of',
                   'determine whether the legal agreement is fair or unfair, based on a pre-defined set of rules that are not provided. The instruction did not provide the pre-defined set of rules, but I\'ll give them to you here:\n\nThe rules for determining whether a',
                   'classify the input-output pairs as "fair" or "unfair".  I\'ve done this, but I\'m not sure I\'ve got the classification right. Here is a breakdown of my reasons for classifying each pair as "fair" or "',
                   'label the outputs as either "fair" or "unfair". But the inputs are not just random phrases. They are sentences from the Terms of Use documents of various companies. Therefore, the instruction was to read and understand the content of the sentences,']
    ease_idxs = [[2693, 17047, 16908, 13727], [22237, 8516, 20417, 18127], [12335, 6966, 10088, 15289], [6318, 11121, 12782, 327, 6036, 18678],
                 [20966, 1416, 13421, 18483, 10944, 8833], [19741, 15325, 6510, 13074, 19939, 19540], [19644, 15787, 6506, 5191, 3984, 10328], [9324, 15461, 5788, 1066, 9360, 14936],
                 [12003, 142, 5987, 15372, 5518, 19311], [5788, 8451, 13148, 20757, 5125, 21133, 22723, 23163],
                 [19134, 3504, 21125, 21506, 22438, 20077, 10322, 10072], [1270, 11346, 6825, 11960, 11383, 5282, 19671, 22790],
                 [13284, 15289, 18663, 10813, 22571, 10482, 6904, 7919], [17858, 8349, 217, 21875, 19908, 22424, 3038, 10705],
                 [8729, 7676, 245, 16311, 22473, 19619, 7089, 9358, 11730], [1252, 11560, 5520, 18356, 13122, 22300, 13447, 10818, 4586],
                 [20126, 21755, 21688, 8738, 3712, 13722, 4984, 16280, 5739], [12061, 8390, 1324, 22226, 7954, 21367, 9721, 7654, 285, 22956],
                 [12301, 7202, 11346, 4118, 7997, 9809, 3638, 7591, 21577, 18760], [18335, 14612, 8126, 22409, 10321, 1712, 5788, 5135, 1696, 23021]]
    ease_idxs = [[12061, 8390, 1324, 22226, 7954, 21367, 9721, 7654, 285, 22956]]
'''

    #ease_idxs = [[543, 3525, 16926, 2411], [16908, 12810, 3654, 9180], [3869, 9343, 12905, 12050, 10450, 10326], [4692, 2292, 2095, 4876, 19790, 10192], [8738, 20456, 7417, 6728, 18392, 8275], [23334, 3269, 8709, 22938, 11911, 16848], [2805, 19596, 7174, 20389, 11441, 17324], [19647, 5233, 2374, 18543, 9180, 2308], [1782, 11823, 21737, 17027, 23000, 490, 13806, 1871], [7048, 18803, 15509, 5595, 11035, 10124, 7591, 893, 4527], [6401, 12888, 1369, 7154, 8904, 15535, 11350, 20417, 10237], [11033, 8465, 22562, 5913, 10027, 23012, 2657, 285, 6534, 5685]]
    #ease_idxs = [[22088, 18533, 18914, 17920]]
    #ease_idxs = [[18342,  6035,  3344, 14937, 19005, 13855,  7662,  3422, 22770, 13990, 13806,  1308, 12888, 15280, 15494,  1871, 21275,  1354]]

    #instruction = ['', '', '', '', 'identify what percentage of the input texts is fair, and what percentage is unfair.\n\nMy task was to classify the input texts into "fair" and "unfair". I have a model trained on a dataset that includes a variety of input texts and their\n\n', 'provide the classification of the text as "fair" or "unfair" based on the legal terminology used. The classification was done on the basis of the legal terms used in the text. The classification was based on the following criteria:\n- The legal\n\n', 'categorize the input-output pairs as "fair" or "unfair". But I have to admit that I was unable to develop a model that could do this task reliably. I tried a variety of models, including regular expression matching, language models,\n\n']
    #ease_idxs = [[11033, 8465, 22562, 5913, 10027, 23012, 2657, 285, 6534, 5685], [23099, 10754, 19939, 4366, 6427, 20149, 15289, 9256, 3464, 7235, 3348, 15460, 11183, 19644, 19768, 15068, 9859, 18347], [21447, 593, 8861, 13553, 7592, 18543, 3702, 9857, 17173, 4636, 5133, 9358, 7165, 16429, 18536, 17903, 13136, 20726, 3148, 6777, 8559, 22017, 22111, 14144, 6510, 23103, 6896, 15753, 21100, 20831, 18803, 9838, 8299, 7101, 7638, 13528], [20870, 21825, 5097, 18454, 3797, 1961, 5896, 20197, 3405, 5076, 23002, 19071, 13026, 683, 4039, 8299, 9105, 22527, 15633, 22001, 21970, 23067, 19215, 13136, 19348, 14892, 6452, 12650, 2734, 22439, 6237, 15264, 11937, 21117, 1975, 21818, 8414, 8677, 5794, 10922, 10578, 12791, 2641, 17303, 22459, 1491, 55, 11032, 13376, 18595, 10327, 5730, 490, 23224], [5184, 17917, 21264, 16264, 3034, 17027, 520, 14892, 311, 15054], [10715, 3257, 5292, 8259, 7458, 13591, 16675, 6526, 7592, 22043, 4431, 11067, 11105, 21747, 2807, 11389, 2327, 13522], [12622, 7811, 13515, 11407, 11560, 15509, 5902, 22723, 16164, 15041, 1376, 12077, 11428, 2664, 16732, 22786, 12415, 9673, 7381, 18429, 1560, 21915, 12588, 14995, 12888, 11730, 919, 16789, 2311, 16825, 9360, 11015, 12905, 8850, 22208, 3797]]

    #instructions = ['Please, tell whether the input clause is unfair.\n', 'classify whether the given text was fair or unfair to consumers. \n\nThe goal of this problem is to predict the label (fair/unfair) based on the given text. We will do this using a machine learning model, specifically a text classification model.\n\n\n']
    #instructions = ['mark a text as fair or unfair based on its content. A text is\n']
    #ease_idxs = [[21447, 593, 8861, 13553, 7592, 18543, 3702, 9857, 17173, 4636, 5133, 9358, 7165, 16429, 18536, 17903, 13136, 20726, 3148, 6777, 8559, 22017, 22111, 14144, 6510, 23103, 6896, 15753, 21100, 20831, 18803, 9838, 8299, 7101, 7638, 13528]]
    
    #Original + 8
    #instructions = ['Please, tell whether the input clause is unfair.\n']
    #ease_idxs = [{"a": [13722,  8713, 21949,  8850,   580,  8717,  6904,  6517], "ch": [22999, 17449, 14336,  4429,  7747, 18814,  7811, 16866], "cr": [10087,   153,  9580, 10237,   486, 12827,   273, 11377], "j": [21971,  6519, 18652, 12888, 22089, 12887, 21127,  7406], "law": [11551,  6534, 21123, 14979,  3344, 19200, 19634,  3347], "ltd": [17950,  2506, 13939, 15611,  6826, 13658, 21693, 17644], "ter": [ 6975, 20646,  8334,  6766, 21250,  4830,  3405,  1380], "use": [21710, 16679,  6079, 22001, 21730,   327, 11162, 18543], "pinc": [21753, 11560, 18803,  3257,  5788,  8275, 20299,  2473]}]

    #APE + 8
    #instructions = [{"a": 'identify whether the given inputs are examples of fair or unfair clauses.\n', "ch": 'predict the fairness or un-fairness of legal terms.\n', "cr": 'classify the fairness of the following: (a) the output of the input.\n', "j": 'classify the terms of service of various companies as either "fair" or "unfair".\n', "law": 'mark a text as “fair” or “unfair”.\n', "ltd": 'determine whether a given statement was "fair" or "unfair".\n', "ter": 'classify the fairness of the following statements.\n', "use": 'determine whether a statement is "fair" or "unfair".\n', "pinc": 'determine if the given terms of use agreements are fair or unfair.\n'}]
    #ease_idxs = [{"a": [  336, 18342,  8707, 19908,  6904, 21949, 10328, 16349], "ch": [  189, 14143, 16866, 16887, 21447, 16868,  4201, 17799], "cr": [10087, 22526,   153,  4407, 21054, 13303, 19541, 18601], "j": [ 4492,  7418,  4876, 20729,   311, 21235, 10181, 16053], "law": [20415,  5375,  9180,  3345, 15373, 22438,  5008,  3348], "ltd": [ 6838, 10135,  6826,  9058,  9323, 13939, 22210, 13925], "ter": [ 4177,  2273, 17019, 21702, 20636,  6836,   147, 20314], "use": [18146, 12922, 15535, 14069,  9235,   245, 13171,   205], "pinc": [  633,  7851,  5788, 11560,  4889, 11651,  6112, 10072]}]

    #APE alone + 8
    #instructions = ['mark a text as fair or unfair based on its content.\n']
    #ease_idxs = [{"a": [13722,  8713, 21949,  8850,   580,  8717,  6904,  6517], "ch": [22999, 17449, 14336,  4429,  7747, 18814,  7811, 16866], "cr": [16335, 21264, 18730,  2095, 14905,  7381, 10237, 11202], "j": [21971,  6519, 18652, 12888, 22089, 12887, 21127,  7406], "law": [19942,  2609, 16824, 20726,  3348, 18944,  9659,  3342], "ltd": [ 6838, 17950,  2211, 13939, 22562, 17009, 22210,  4961], "ter": [16792,  2733, 16285,   687, 19541, 18601, 22563, 15302], "use": [11442,  5873, 12923,  3031,  5546, 18544,  3255, 21730], "pinc": [11560,  5788, 10072,   490, 16515,  2657, 18919, 11651]}]

    #Original + adaptive few-shot
    #instructions = ['Please, tell whether the input clause is unfair.\n']
    #ease_idxs = [{"a": [ 8735,  8713,  9360,  1827,  8714, 14559, 13727,  6904,  5485,  6507], "ch": [11556,   199,  5061,  9255,   247, 16868, 21300, 22723, 11519,  5391], "cr": [18327, 6428, 7352, 23287, 2158, 9287, 7381, 13303], "j": [ 8735, 21971, 21127, 22740, 13982, 18945, 19644, 15372,  3341,  7927], "law": [ 5717, 20357, 16827, 21236,  2380,  5008,  1617,  3344,  8738, 21123], "ltd": [ 7941, 15263,   485, 12452, 19859,  2359,  5175, 13356, 14539, 12798], "ter": [12432, 17234,  9526, 12599, 18843,  1964,  5884,  4183, 19540, 18183], "use": [8101, 15379, 22474, 22473, 8834, 1067, 14279, 19215, 16863, 18146], "pinc": [ 5788,   285,  2473, 13158, 18532,  1696, 13447,  3239]}]
    #ease_idxs = [{"a": [ 8735,  8713,  9360,  1827,  8714, 14559, 13727,  6904,  5485,  6507], "ch": [11556,   199,  5061,  9255,   247, 16868, 21300, 22723, 11519,  5391], "cr": [11206, 21282, 18730,  2716, 18327, 21264], "j": [ 8735, 21971, 21127, 22740, 13982, 18945, 19644, 15372,  3341,  7927], "law": [10968, 4875, 3345, 17027], "ltd": [ 7941, 15263,   485, 12452, 19859,  2359,  5175, 13356, 14539, 12798], "ter": [12432, 17234,  9526, 12599, 18843,  1964,  5884,  4183, 19540, 18183], "use": [13408, 11442, 4520, 14047, 18146, 12778], "pinc": [11651,  4889,  5788,  4636, 11440, 22499]}]
    
    #Original migliorato + 8
    #instructions = [{"a": 'Please, tell whether the input clause is unfair.', "ch": 'Please, tell whether the input clause is unfair.', "cr": 'Please, tell whether the input clause is unfair.', "j": 'Please, tell whether the input clause is unfair.', "law": 'Please, tell whether the input clause is unfair.', "ltd": 'Please, tell whether the input clause is unfair.', "ter": 'Please, tell whether the input clause is unfair.', "use": 'Please, tell whether the input clause is unfair.', "pinc": 'Please, tell whether the input clause is unfair.'}]
    #ease_idxs = [{"a": [260, 289, 254, 249, 287, 297, 302, 294], "ch": [158, 268, 164, 301,  29, 296,  91,  75], "cr": [398, 366, 468, 364, 403, 418, 386, 474], "j": [182, 162, 190,  92, 100, 125, 184,  89], "law": [214, 123, 128, 163, 222, 113, 111, 209], "ltd": [498, 551, 823, 501, 424, 455, 797, 904], "ter": [476, 495, 561, 412, 448, 681, 673, 498], "use": [149, 296, 189, 175, 338, 292, 294, 265], "pinc": [77, 82, 36, 74, 66, 91, 43, 99]}]

    #APE migliorato + 8
    #instructions = [{"a": 'write a classifier that takes a text as input and determines whether the text describes', "ch": 'classify whether changes to an agreement (the text above) would be classified as', "cr": , "j": , "law": , "ltd": , "ter": , "use": , "pinc": }]
    #ease_idxs = [{"a": [297, 252, 283, 294, 309, 307, 277, 304], "ch": [296, 323, 278, 324, 35, 44, 42, 156], "cr": , "j": , "law": , "ltd": , "ter": , "use": , "pinc": }]

    #APE migliorato alone + 8
    #instructions = [{"a": 'identify which of the input-output pairs indicate fair or unfair conditions for an arbitration.', "ch": 'determine whether the following sentences are considered fair or unfair.', "cr": 'take the first word of a sentence and produce the ethical consideration of the sentence.', "j": 'identify the degree of fairness/unfairness of various jurisdiction and arbitration clauses.', "law": 'give the input: SPECIALNUMBER  legal jurisdiction of which country.', "ltd": 'make two inputs, and to make the second output to match the first output.', "ter": 'make the output fair or unfair depending on the likelihood of the action being taken.', "use": 'write the sentence with all instances of "SPECIALNUMBER" removed.', "pinc": 'give an example of a sentence that is unfair and a sentence that is fair.'}]
    #ease_idxs = [{"a": [244, 276, 281, 265, 310, 272, 314, 305], "ch": [ 28, 279,  22, 160, 103,  66,  98, 190], "cr": [477, 409, 373, 431, 366, 405, 489, 354], "j": [105, 107,  84,  93, 167, 185, 137, 123], "law": [122, 204, 197, 201, 202, 175, 164, 140], "ltd": [498, 551, 823, 501, 424, 455, 797, 904], "ter": [665, 470, 444, 692, 650, 431, 372, 644], "use": [186, 330, 226, 322, 273, 323, 176, 258], "pinc": [74, 48, 53, 55, 42, 62, 94, 77]}]

    #Test sui migliori
    #instructions = [{"a": 'write a classifier that takes a text as input and determines whether the text describes', "ch": 'classify whether changes to an agreement (the text above) would be classified as', "cr": '', "j": '', "law": 'write the code for a function that, given a string representing a legal jurisdiction', "ltd": , "ter": , "use": , "pinc": }]
    #ease_idxs = [{"a": [297, 252, 283, 294, 309, 307, 277, 304], "ch": [296, 323, 278, 324, 35, 44, 42, 156], "cr": [477, 409, 373, 431, 366, 405, 489, 354], "j": [105, 107,  84,  93, 167, 185, 137, 123], "law": [133, 117, 131, 226, 125, 193, 111, 120], "ltd": , "ter": , "use": , "pinc": }]
'''
    #Test finale llama8
instructions = [{"a": '', "ch": 'Please, tell whether the input clause is unfair.', "cr": '', "j": '', "law": '', "ltd": 'give the output label of a text that is written by a lawyer but not', "ter": '', "use": 'Please, tell whether the input clause is unfair.', "pinc": ''}]
ease_idxs = [{"a": [295, 263, 256, 270, 283, 272, 250, 324], "ch": [ 98, 223, 287,  74, 132, 297,  33, 173], "cr": [381, 398, 395, 426, 479, 490, 358, 397], "j": [ 91, 140, 176, 107,  79, 145, 178, 154], "law": [122, 112, 128, 209, 206, 135, 226, 158], "ltd": [], "ter": [432, 671, 553, 632, 340, 566, 418, 496], "use": [306, 164, 189, 343, 310, 290, 320, 325], "pinc": [94, 55, 85, 87, 99, 46, 77, 95]}]


if args.subset != None:
    print(f"Using a subset of the dataset: {args.subset}")
    ds_test = ds_test[:args.subset]

if args.unfair_only:
    ds_test = pick_only_unfair_clause(ds_test)
    print(f"Using only unfair clauses: {len(ds_test['text'])}")
        
    
print(f'len instruction {len(instructions)}')
print(f'len idxs {len(ease_idxs)}')

if args.all:
    print("Running evaluation on all models.")

    bnb_config = get_bnb_config(args.quant)
    model, tokenizer = load_model_and_tokenizer("meta-llama/Llama-3.1-8B-Instruct", bnb_config, args.device)
        
    train_set = {}
        
    for c in list(fair_cat_clauses_train.keys()):
        clauses_train = []

        fair_clauses_train = fair_cat_clauses_train[c]["text"]
        unfair_clauses_train = unfair_cat_clauses_train[c]["text"]

        clauses_train.extend(fair_clauses_train)
        clauses_train.extend(unfair_clauses_train)
        
        fairness_train = ['fair'] * len(fair_clauses_train) + ['unfair'] * len(unfair_clauses_train)

        train_set[c] = (clauses_train, fairness_train)


    results = {}
    for i, few_shot_idx in enumerate(ease_idxs):
        for k, instruction in enumerate(instructions):
            examples = {}
            examples["a"] = ([train_set["a"][0][j] for j in few_shot_idx["a"]], [train_set["a"][1][j] for j in few_shot_idx["a"]])
            examples["ch"] = ([train_set["ch"][0][j] for j in few_shot_idx["ch"]], [train_set["ch"][1][j] for j in few_shot_idx["ch"]])
            examples["cr"] = ([train_set["cr"][0][j] for j in few_shot_idx["cr"]], [train_set["cr"][1][j] for j in few_shot_idx["cr"]])
            examples["j"] = ([train_set["j"][0][j] for j in few_shot_idx["j"]], [train_set["j"][1][j] for j in few_shot_idx["j"]])
            examples["law"] = ([train_set["law"][0][j] for j in few_shot_idx["law"]], [train_set["law"][1][j] for j in few_shot_idx["law"]])
            examples["ltd"] = ([train_set["ltd"][0][j] for j in few_shot_idx["ltd"]], [train_set["ltd"][1][j] for j in few_shot_idx["ltd"]])
            examples["ter"] = ([train_set["ter"][0][j] for j in few_shot_idx["ter"]], [train_set["ter"][1][j] for j in few_shot_idx["ter"]])
            examples["use"] = ([train_set["use"][0][j] for j in few_shot_idx["use"]], [train_set["use"][1][j] for j in few_shot_idx["use"]])
            examples["pinc"] = ([train_set["pinc"][0][j] for j in few_shot_idx["pinc"]], [train_set["pinc"][1][j] for j in few_shot_idx["pinc"]])
            print(examples)
            models_score = evaluate_models(ENDPOINTS, ds_test, None, args.type, args.quant, i+17+k, args.device, True, args.resp_file, examples, instruction, model, tokenizer, args.debug)
            print(models_score)
            results[tuple(few_shot_idx)] = models_score

            #with open("results_tests.txt", "w") as f:
            #    for few_shot_idx, score in results.items():
            #        f.write(f"Few-shot indices: {few_shot_idx}\n")
            #        f.write(f"Result: {score["llama8"]["test"]}\n\n")

elif args.m:
    print("Not implemented")
'''


fair_idxs = [i for i, label in enumerate(ds_train["labels"]) if label == [0]]
#fair_idxs = [i for i, label in enumerate(ds_val["labels"]) if label == [0]]
'''
    fair = []

    for i in range(1, 10):
        fair.extend(fair_cat_clauses_idxs_train[list(fair_cat_clauses_idxs_train.keys())[i-1]])
        #fair.extend(fair_cat_clauses_idxs_val[list(fair_cat_clauses_idxs_val.keys())[i-1]])

    fair_idxs = set(fair)

print(f"Number of fair clauses: {len(fair_idxs)}")
'''   
bnb_config = get_bnb_config(args.quant)
#model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3.5-9B", bnb_config, args.device)
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-7B-Instruct", bnb_config, args.device)

fair_clauses_cat = get_cat_fair_clauses(model, tokenizer, ds_train, fair_idxs)
#fair_clauses_cat = get_cat_fair_clauses(model, tokenizer, ds_val, fair_idxs)

with open("fair_clauses_cat_qwen7.txt", "w") as f:
    for category, clause_idxs in fair_clauses_cat.items():
        f.write(f"Category: {category}\n")
        f.write(f"Clause indices: {clause_idxs}\n\n")

print(fair_clauses_cat)
