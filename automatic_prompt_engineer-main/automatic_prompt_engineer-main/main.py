from automatic_prompt_engineer import ape
from huggingface_hub import login
from automatic_prompt_engineer import data, llm, config

ds_train = data.load_dataset("./142_dataset/tos.hf/", split="train")
ds_val = data.load_dataset("./142_dataset/tos.hf/", split="validation")
ds_test = data.load_dataset("./142_dataset/tos.hf/", split="test")


fair_cat_clauses_idxs_train = {'a': [225, 339, 556, 557, 562, 563, 569, 584, 1014, 1134, 1204, 1450, 1676, 1807, 1815, 1818, 1820, 1824, 1828, 1832, 1847, 1848, 2304, 2305, 2311, 2313, 2317, 2324, 2327, 2335, 2343, 2802, 2806, 3015, 3121, 3180, 3207, 3208, 4274, 4287, 4300, 4308, 4464, 4474, 4475, 4477, 4482, 4485, 4496, 4664, 4666, 4688, 5376, 5484, 5486, 5487, 5493, 5501, 5589, 5723, 5736, 5738, 5740, 5743, 5757, 5814, 5983, 5985, 5986, 5991, 5996, 6001, 6009, 6011, 6017, 6018, 6021, 6028, 6029, 6033, 6173, 6380, 6416, 6496, 6503, 6511, 6718, 6725, 7976, 7978, 8000, 8030, 8058, 8085, 8096, 8103, 8122, 8136, 8155, 8164, 8166, 8168, 8189, 8226, 8228, 8262, 8690, 8691, 8693, 8699, 8700, 8711, 8712, 8723, 8831, 8846, 8852, 8855, 8859, 8865, 8910, 9238, 9351, 9361, 9363, 9971, 9984, 10183, 10314, 10320, 10377, 10533, 10537, 10541, 10543, 10547, 10549, 10550, 10966, 10970, 10978, 10981, 10983, 10986, 10987, 10991, 11256, 11258, 11260, 11262, 11267, 11639, 11658, 11913, 11934, 11935, 11940, 11945, 11946, 11958, 13364, 13368, 13369, 13371, 13525, 13527, 13720, 13723, 13728, 13730, 13732, 13735, 13737, 13740, 13746, 13747, 14273, 14275, 14280, 14555, 14572, 14575, 14589, 14591, 14593, 14751, 14831, 14836, 14839, 14895, 14903, 14907, 14912, 14925, 14962, 14966, 14972, 14977, 14984, 15140, 15343, 15777, 15819, 15960, 16101, 16190, 16345, 16362, 16365, 16371, 16372, 16377, 16382, 16384, 16393, 16677, 16911, 18371, 18373, 18710, 19216, 19219, 19263, 19577, 19578, 19589, 19593, 19594, 19597, 19604, 19605, 19620, 19623, 19668, 19808, 19902, 19907, 19909, 19911, 19914, 19916, 19917, 19920, 19931, 20278, 21013, 21677, 21758, 21948, 21954, 22314, 22437, 22446, 22448, 22450, 22453], 'ch': [1134, 3708, 4200, 5589, 5806, 5929, 6895, 7052, 7543, 9830, 10593, 11031, 14820, 15280, 15926, 15928, 16371, 18809, 21079, 21230, 21376, 21591], 'cr': [132, 191, 274, 275, 417, 418, 424, 425, 434, 438, 462, 796, 867, 1059, 1082, 1134, 1204, 1293, 1382, 1450, 1667, 1706, 1709, 1730, 1926, 2024, 2137, 2152, 2154, 2190, 2210, 2233, 2407, 2413, 2515, 2535, 2593, 2638, 2640, 2696, 2697, 2699, 2710, 2711, 2872, 2985, 3001, 3079, 3121, 3122, 3127, 3142, 3265, 3286, 3303, 3313, 3318, 3320, 3368, 3369, 3375, 3380, 3410, 3425, 3428, 3451, 3452, 3469, 3536, 3573, 3680, 4300, 4436, 4516, 4517, 4664, 4666, 4688, 5137, 5138, 5578, 5589, 5598, 5638, 5644, 5650, 5783, 5805, 5893, 5920, 5926, 6159, 6173, 6176, 6214, 6235, 6236, 6242, 6378, 6380, 6407, 6416, 6422, 6449, 6550, 6567, 6775, 6792, 6800, 6928, 6973, 6986, 7010, 7024, 7027, 7096, 7327, 7330, 7353, 7371, 7380, 7521, 7670, 7772, 7853, 7976, 7978, 7987, 8000, 8020, 8030, 8058, 8065, 8085, 8096, 8103, 8122, 8136, 8155, 8164, 8166, 8189, 8214, 8226, 8228, 8262, 8304, 8327, 8351, 8454, 8456, 8461, 8465, 8477, 8478, 8518, 8533, 8536, 8549, 8552, 8600, 8693, 8723, 8787, 8802, 8803, 9018, 9092, 9098, 9100, 9146, 9217, 9238, 9277, 9278, 9285, 9293, 9296, 9541, 9546, 9946, 9971, 9984, 10045, 10061, 10062, 10063, 10083, 10168, 10183, 10188, 10226, 10227, 10229, 10240, 10422, 10582, 10639, 10808, 10898, 11019, 11085, 11086, 11115, 11203, 11336, 11338, 11433, 11463, 11621, 11725, 11727, 11731, 11732, 11742, 11765, 11789, 12122, 12160, 12223, 12316, 12477, 12685, 12817, 12818, 12819, 12820, 13002, 13006, 13007, 13055, 13091, 13104, 13272, 13273, 13396, 13494, 13746, 13784, 14022, 14023, 14112, 14145, 14235, 14251, 14353, 14371, 14391, 14403, 14436, 14673, 14730, 14786, 14820, 14831, 14836, 14839, 14881, 14895, 14903, 14904, 14907, 14912, 14925, 14962, 14966, 14972, 14977, 14984, 15114, 15140, 15142, 15160, 15257, 15334, 15343, 15494, 15546, 15584, 15655, 15698, 15789, 15791, 15798, 15819, 15825, 15873, 15883, 15885, 15938, 15960, 15977, 16019, 16092, 16101, 16190, 16205, 16371, 16540, 16665, 16677, 16702, 16703, 16746, 16846, 16910, 16939, 16978, 17042, 17046, 17080, 17113, 17169, 17171, 17702, 18216, 18217, 18273, 18331, 18453, 18528, 18565, 18710, 18883, 18885, 18961, 19213, 19249, 19334, 19511, 19664, 19685, 19760, 19769, 19770, 19782, 19847, 19869, 19931, 20278, 20360, 20388, 20395, 20761, 20766, 20906, 21343, 21370, 21411, 21443, 21449, 21476, 21492, 21509, 21560, 21576, 21591, 21606, 21639, 21677, 21717, 21758, 21793, 21838, 21905, 21919, 22021, 22062, 22275, 22287, 22305, 22329, 22358, 22543, 22658, 22704, 22799, 23105, 23174, 23225, 23234, 23250, 23277], 'j': [214, 541, 595, 1622, 1807, 2295, 2304, 2349, 2591, 2802, 2816, 2817, 3219, 4309, 4359, 5486, 7379, 7593, 7659, 7924, 8734, 8747, 8866, 9150, 10338, 10347, 11104, 11107, 11914, 11919, 11922, 11923, 11926, 11927, 12628, 13124, 13363, 13373, 13375, 13385, 13530, 13748, 13760, 14304, 14560, 14570, 14801, 15766, 16045, 16371, 16834, 17031, 17073, 17323, 17675, 17679, 17680, 18109, 18371, 18943, 19202, 19263, 19265, 19299, 19611, 19633, 19645, 19646, 19647, 21126, 21234, 21323, 21816, 21956, 22452, 23222], 'law': [214, 307, 541, 1134, 1450, 1616, 1637, 1812, 2333, 2816, 2872, 3015, 3180, 3740, 4309, 4350, 4497, 4664, 4666, 4688, 5007, 5589, 6535, 7074, 7080, 7145, 7405, 7589, 7593, 7653, 7659, 7922, 7976, 7978, 8000, 8030, 8058, 8085, 8096, 8103, 8122, 8136, 8155, 8164, 8166, 8168, 8189, 8226, 8228, 8262, 8304, 8690, 8737, 8749, 8846, 10180, 10338, 10527, 10552, 10966, 10970, 11104, 11256, 11271, 11298, 11639, 11913, 11914, 11919, 11922, 11923, 11926, 11927, 12314, 12625, 12628, 12891, 12940, 13135, 13380, 13715, 13760, 14022, 14023, 14036, 14201, 14303, 14552, 14697, 14720, 14740, 14818, 14831, 14836, 14839, 14895, 14903, 14907, 14912, 14925, 14962, 14966, 14972, 14977, 14978, 14984, 15766, 15777, 16045, 16051, 16517, 16519, 16521, 16834, 17023, 17026, 17031, 17326, 17675, 17980, 18339, 18480, 18710, 18943, 19265, 19633, 19637, 19940, 20185, 20356, 20414, 20475, 20725, 21234, 21323, 21816, 21969, 22437, 22650, 23222], 'ltd': [177, 274, 297, 320, 360, 472, 527, 532, 551, 589, 612, 1134, 1521, 1537, 1549, 1575, 1578, 1667, 1755, 1758, 1763, 1864, 1980, 1981, 2090, 2096, 2107, 2113, 2202, 2254, 2338, 2350, 2356, 2401, 2407, 2441, 2601, 2640, 2781, 2785, 2788, 2797, 2871, 2963, 2985, 3094, 3166, 3178, 3189, 3192, 3194, 3332, 3452, 3506, 3661, 3671, 3675, 3679, 3681, 3749, 3923, 3961, 4025, 4029, 4093, 4223, 4360, 4430, 4451, 4455, 4461, 4576, 4581, 4582, 4595, 4604, 4611, 4664, 4666, 4688, 4697, 4767, 4848, 4940, 4954, 4956, 5159, 5312, 5314, 5315, 5316, 5319, 5332, 5476, 5501, 5556, 5564, 5589, 5648, 5672, 5673, 5677, 5678, 5682, 5684, 5685, 5813, 5857, 5973, 5975, 5976, 5981, 6043, 6465, 6468, 6476, 6482, 6484, 6487, 6550, 6714, 6810, 6843, 6847, 6870, 6873, 6884, 6928, 7067, 7143, 7283, 7400, 7401, 7458, 7538, 7556, 7557, 7618, 7635, 7636, 7649, 7670, 7798, 7872, 7878, 7881, 7976, 7978, 8000, 8030, 8058, 8085, 8096, 8103, 8122, 8136, 8155, 8164, 8166, 8174, 8179, 8182, 8189, 8226, 8228, 8262, 8315, 8372, 8381, 8383, 8430, 8531, 8545, 8560, 8567, 8631, 8659, 8665, 8667, 8668, 8670, 8673, 8816, 9035, 9044, 9099, 9109, 9172, 9318, 9319, 9440, 9599, 9736, 10152, 10188, 10272, 10275, 10294, 10302, 10509, 10513, 10541, 10562, 10582, 10659, 10660, 10666, 10672, 10673, 10727, 10735, 10750, 10808, 10886, 10901, 10907, 10909, 10918, 10931, 11019, 11084, 11092, 11097, 11099, 11249, 11253, 11336, 11338, 11391, 11400, 11410, 11433, 11462, 11724, 11735, 11798, 11865, 11871, 11874, 11892, 11893, 11912, 12219, 12242, 12571, 12796, 12802, 12805, 12813, 12838, 12840, 12842, 12938, 13080, 13099, 13104, 13111, 13113, 13115, 13116, 13120, 13344, 13355, 13388, 13396, 13444, 13520, 13542, 13653, 13714, 13716, 13784, 13912, 14034, 14154, 14160, 14178, 14180, 14181, 14326, 14367, 14416, 14524, 14526, 14535, 14635, 14731, 14813, 14831, 14836, 14839, 14873, 14887, 14893, 14895, 14903, 14907, 14912, 14925, 14939, 14962, 14966, 14972, 14977, 14984, 15249, 15258, 15344, 15348, 15349, 15394, 15438, 15460, 15606, 15685, 15712, 16008, 16062, 16068, 16070, 16325, 16404, 16444, 16534, 16537, 16542, 16665, 16800, 16812, 16815, 16837, 16991, 16992, 16996, 17007, 17012, 17208, 17250, 17252, 17293, 17312, 17580, 17584, 17592, 17595, 17620, 17649, 17760, 18069, 18081, 18097, 18134, 18212, 18242, 18246, 18304, 18344, 18361, 18381, 18556, 18579, 18583, 18710, 18731, 18798, 18827, 18890, 18934, 18937, 18953, 19175, 19213, 19261, 19552, 19556, 19557, 19559, 19561, 19567, 19664, 19801, 19881, 19886, 19977, 19981, 20126, 20278, 20382, 20408, 20422, 20437, 20695, 20706, 20906, 21073, 21212, 21215, 21218, 21222, 21321, 21500, 21620, 21647, 21666, 21669, 21758, 21824, 21859, 21919, 21927, 21936, 22069, 22071, 22073, 22208, 22353, 22402, 22483, 22504, 22580, 22583, 22598, 22604, 22606, 22614, 22702, 22710, 22732, 22972, 23120, 23225, 23226, 23238, 23257, 23319, 23324, 23325, 23327], 'ter': [126, 139, 143, 267, 303, 480, 796, 860, 970, 1134, 1204, 1382, 1391, 1400, 1426, 1450, 1667, 1701, 2067, 2218, 2366, 2369, 2407, 2425, 2518, 2585, 2640, 2686, 2696, 2872, 2985, 3013, 3114, 3121, 3123, 3125, 3127, 3132, 3138, 3142, 3175, 3196, 3363, 3586, 3706, 4137, 4174, 4182, 4300, 4342, 4422, 4437, 4439, 4440, 4446, 4516, 4628, 4631, 4664, 4666, 4669, 4688, 4923, 4998, 5004, 5305, 5589, 5598, 5644, 5783, 5844, 5893, 5909, 6165, 6173, 6175, 6235, 6300, 6369, 6380, 6382, 6409, 6416, 6418, 6420, 6422, 6550, 6636, 6831, 6907, 6928, 6948, 6986, 7024, 7096, 7098, 7099, 7100, 7380, 7567, 7568, 7571, 7656, 7670, 7749, 7888, 7893, 7895, 7916, 7976, 7978, 8000, 8030, 8042, 8058, 8085, 8096, 8103, 8122, 8136, 8155, 8157, 8162, 8164, 8166, 8189, 8226, 8228, 8262, 8270, 8304, 8332, 8420, 8600, 8607, 8635, 8693, 8723, 8802, 8844, 8879, 8910, 9015, 9125, 9146, 9238, 9254, 9515, 9518, 9779, 9782, 9899, 9902, 9984, 10034, 10052, 10154, 10157, 10159, 10183, 10188, 10226, 10277, 10399, 10500, 10504, 10582, 10699, 10808, 10895, 10898, 11019, 11086, 11090, 11115, 11336, 11338, 11413, 11433, 11731, 11866, 12017, 12036, 12061, 12075, 12127, 12168, 12189, 12205, 12258, 12259, 12278, 12282, 12285, 12290, 12306, 12313, 12320, 12477, 12478, 12547, 12596, 12601, 12607, 12609, 12942, 13002, 13092, 13148, 13200, 13396, 13531, 13538, 13539, 13568, 13685, 13766, 13784, 13880, 13967, 14019, 14022, 14023, 14035, 14093, 14099, 14190, 14403, 14421, 14598, 14743, 14831, 14836, 14839, 14895, 14903, 14907, 14908, 14912, 14925, 14962, 14966, 14972, 14977, 14984, 15026, 15140, 15145, 15151, 15160, 15191, 15199, 15214, 15215, 15257, 15303, 15337, 15341, 15343, 15355, 15693, 15707, 15789, 15791, 15811, 15819, 15821, 15823, 15825, 15853, 15873, 15953, 15960, 15962, 15976, 16019, 16023, 16086, 16092, 16101, 16108, 16111, 16127, 16173, 16190, 16194, 16197, 16205, 16237, 16320, 16341, 16371, 16378, 16665, 16677, 16938, 16978, 17015, 17113, 17171, 17223, 17224, 17227, 17235, 17614, 17702, 17723, 18025, 18185, 18216, 18273, 18318, 18323, 18324, 18477, 18710, 18769, 18883, 19188, 19213, 19259, 19534, 19658, 19664, 19869, 19900, 19931, 19988, 20009, 20024, 20278, 20313, 20344, 20791, 20792, 20849, 20850, 21039, 21083, 21280, 21301, 21353, 21396, 21660, 21677, 21689, 21701, 21703, 21719, 21722, 21758, 21864, 21871, 21872, 21924, 22021, 22060, 22062, 22266, 22268, 22287, 22305, 22407, 22493, 22572, 22704, 22799, 22928, 23161, 23172, 23225], 'use': [116, 251, 340, 377, 387, 458, 1068, 1667, 1690, 1864, 1865, 2067, 2229, 2407, 2519, 2585, 2640, 2672, 2985, 3026, 3358, 3363, 4367, 4696, 4697, 4891, 4901, 5047, 5192, 5431, 5549, 5588, 5624, 5642, 5790, 5881, 6550, 6702, 6928, 6944, 6968, 7096, 7110, 7151, 7401, 7670, 7748, 8261, 8270, 8287, 8396, 8407, 8783, 8835, 9012, 9069, 9378, 9971, 10188, 10197, 10379, 10385, 10582, 10808, 11019, 11020, 11165, 11170, 11336, 11338, 11433, 11606, 11611, 11856, 12031, 12771, 12894, 12970, 12971, 13159, 13396, 13453, 13585, 13784, 13847, 15007, 15613, 16665, 16695, 16850, 16871, 17162, 17367, 17382, 18115, 18131, 18167, 18452, 18503, 18512, 18592, 18646, 18647, 18798, 18799, 19213, 19664, 19684, 19714, 19717, 19727, 19731, 19983, 20371, 20543, 20796, 20858, 20933, 21249, 21312, 21826, 21888, 21997, 23053, 23082, 23127, 23225, 23236, 23280, 23293], 'pinc': [491, 2036, 2376, 2683, 3442, 3505, 4360, 5349, 5642, 6980, 7584, 8197, 8286, 8644, 9377, 9378, 10386, 10409, 11129, 11372, 12395, 13542, 13849, 14045, 14976, 16371, 18751, 18820, 18832, 19096, 19713, 20887, 21374, 21894]}
fair_cat_clauses_idxs_val = {'a': [265, 385, 591, 767, 1018, 1019, 1250, 1251, 1252, 1253, 1258, 1265, 1274, 1276, 1281, 1282, 1285, 2601, 2699, 2913, 2930, 2933, 2944, 2948, 2949, 2950, 2969, 3965, 3969, 3976, 3980, 3981, 3983, 4002, 4345, 4370, 4374, 4382, 4389, 4398, 4404, 4405, 4410, 4415, 4513, 4517, 4532, 4969, 4970, 4982, 4984, 4987, 4992, 5005, 5036, 5300, 5314, 5316, 5317, 5322, 5324, 5329, 5330, 5755, 6022, 6023, 6028, 6033, 6042, 6043, 6047, 6049, 6866, 7082, 7161, 7171, 7172, 7175, 7177, 7178, 7179, 7187, 7188, 7192, 7235, 7254, 7420, 7421, 7430, 7432, 7523, 7750, 7751, 7755, 7765, 7778, 8119, 8322, 8538, 8544, 8548, 8558, 8563, 8567, 8570, 8822, 8823, 8826, 8873, 9140, 9147, 9155, 9375, 9378, 9381, 9382, 10183], 'ch': [1508, 2751, 4303, 5363, 6262, 7535], 'cr': [0, 71, 160, 180, 182, 265, 356, 385, 473, 571, 634, 674, 870, 1120, 1161, 1165, 1219, 1890, 2214, 2219, 2222, 2226, 2227, 2430, 2601, 2652, 2658, 2805, 2840, 2889, 2890, 3111, 3157, 3283, 3404, 3682, 3695, 3791, 3792, 3894, 4017, 4082, 4110, 4111, 4209, 4211, 4270, 4311, 4480, 4502, 4509, 4536, 4706, 4760, 4765, 4769, 4770, 4775, 4833, 4865, 5119, 5203, 5219, 5225, 5344, 5356, 5357, 5358, 5359, 5400, 5426, 5432, 5650, 5671, 5744, 5774, 5777, 5986, 6183, 6193, 6194, 6226, 6262, 6263, 6727, 6778, 6840, 6885, 6888, 6893, 6918, 7048, 7078, 7095, 7255, 7369, 7511, 7535, 7542, 7611, 7649, 7662, 7663, 7685, 7778, 7789, 7837, 7870, 7873, 8010, 8030, 8119, 8169, 8312, 8349, 8350, 8372, 8374, 8391, 8400, 8402, 8411, 8469, 8707, 8788, 9033, 9228, 9233, 9235, 9238, 9397, 9489, 9528, 10077, 10129, 10169], 'j': [553, 1250, 2166, 2350, 2930, 3965, 4154, 4159, 4202, 5289, 6022, 6037, 6052, 6275, 6278, 6786, 7082, 7192, 7420, 7440, 7523, 8296, 9136], 'law': [553, 779, 782, 999, 1294, 1776, 2166, 2350, 2512, 2931, 3420, 3782, 3953, 4159, 4202, 4345, 4444, 4969, 5474, 5511, 6017, 6072, 6786, 6787, 7522, 7745, 7747, 7767, 7950, 8296, 8521, 8522, 8819, 9375, 9443, 9444], 'ltd': [0, 124, 125, 160, 265, 278, 382, 385, 387, 402, 413, 538, 693, 752, 757, 760, 762, 1121, 1234, 1235, 1237, 1243, 1408, 1463, 1469, 1621, 1752, 1802, 1988, 2085, 2209, 2255, 2278, 2279, 2284, 2289, 2291, 2294, 2445, 2674, 2681, 2692, 2742, 2795, 2797, 2916, 2925, 2927, 3139, 3359, 3366, 3446, 3577, 3924, 3944, 3945, 4123, 4124, 4127, 4199, 4209, 4211, 4309, 4429, 4430, 4431, 4432, 4435, 4539, 4540, 4541, 4549, 4589, 4596, 4608, 4667, 4954, 4961, 4963, 4994, 5063, 5236, 5248, 5253, 5256, 5259, 5301, 5344, 5459, 5492, 5494, 5562, 5695, 5703, 5718, 5784, 5786, 5829, 5833, 5836, 5842, 5877, 5940, 5949, 5958, 5961, 5989, 5991, 5992, 5998, 6000, 6005, 6006, 6012, 6024, 6256, 6313, 6314, 6482, 6787, 6840, 6864, 7088, 7124, 7127, 7136, 7141, 7144, 7254, 7255, 7406, 7411, 7414, 7515, 7671, 7673, 7706, 7708, 7711, 7735, 7778, 7789, 7913, 7939, 7943, 7988, 8071, 8119, 8139, 8189, 8194, 8216, 8223, 8239, 8240, 8242, 8319, 8501, 8568, 8765, 8769, 8772, 9226, 9230, 9349, 9351, 9357, 9374, 9396, 9397, 9645, 10151], 'ter': [0, 24, 42, 160, 187, 265, 340, 385, 549, 740, 744, 746, 907, 945, 951, 1144, 1305, 1343, 1890, 1982, 2303, 2502, 2601, 2731, 2888, 2910, 2913, 3023, 3169, 3189, 3268, 3339, 3621, 3695, 3835, 3927, 4110, 4115, 4118, 4209, 4211, 4270, 4419, 5003, 5134, 5195, 5272, 5277, 5344, 5514, 5579, 5755, 5889, 6183, 6262, 6290, 6455, 6727, 6840, 6856, 6914, 7060, 7067, 7075, 7255, 7398, 7468, 7473, 7555, 7699, 7778, 7789, 7804, 7846, 8056, 8119, 8194, 8315, 8334, 8336, 8411, 8457, 8463, 8464, 8754, 9283, 9309, 9319, 9324, 9397, 9528, 9815, 9841, 10039, 10169], 'use': [0, 15, 16, 83, 166, 307, 439, 446, 464, 683, 975, 1159, 1345, 1505, 2054, 2159, 2172, 2305, 2613, 3013, 3296, 4140, 4172, 4209, 4211, 4540, 4545, 4571, 5029, 5344, 5350, 5354, 5375, 5617, 5854, 6000, 6042, 6139, 6840, 7254, 7255, 7257, 7262, 7274, 7294, 7320, 7477, 7717, 7789, 7998, 8095, 8382, 8632, 8851, 9176, 9189, 9252, 9263, 9397, 9542, 9543, 10028], 'pinc': [525, 975, 3817, 5029, 5467, 5605, 5818, 7254, 8418, 9514, 9546]}

fair_cat_clauses_train = {}
unfair_cat_clauses_train = {}
fair_cat_clauses_val = {}
unfair_cat_clauses_val = {}

for i, cat in enumerate(fair_cat_clauses_idxs_train.keys()):
    fair_cat_clauses_train[cat] = ds_train.filter(lambda example, index: index in fair_cat_clauses_idxs_train[cat] and len(example['text']) > 0, with_indices=True)
    unfair_cat_clauses_train[cat] = ds_train.filter(lambda example: (i+1) in example["labels"])
    
    fair_cat_clauses_val[cat] = ds_train.filter(lambda example, index: index in fair_cat_clauses_idxs_train[cat], with_indices=True)
    unfair_cat_clauses_val[cat] = ds_train.filter(lambda example: (i+1) in example["labels"])


#clauses_train = ds_train["text"]
#clauses_val = ds_val["text"]
#clauses_test = ds_test["text"]

'''
fairness_train = ['fair' if label == [0] else 'unfair' for label in ds_train["labels"]]
labels_train = ds_train["labels"]
'''
#fairness_val = ['fair' if label == [0] else 'unfair' for label in ds_val["labels"]]
#fairness_test = ['fair' if label == [0] else 'unfair' for label in ds_test["labels"]]

#labels_val = ds_val["labels"]
#labels_test = ds_test["labels"]


print("Token login")
login()

###### APE + eval RIFARE
'''
category_definitions = ["A clause is unfair whenever arbitration is binding and not optional, and/or should take place in a country different from the consumer's place of residence, and/or be based not on law but on other arbitration rules or the arbiter's discretion.", "A clause is unfair when it specifies if and under what conditions the provider can unilaterally change or modify the contract and/or the service", "A clause is unfair if it gives the provider the right to modify, delete, or remove the user's content, including in-app purchases, under specific conditions or at any time, at their full discretion, for any or no reason, with or without notice or the possibility to retrieve the content", "A clause is unfair whenever it states that judicial proceedings take place away (i.e., in a different city or country from the consumer's place of residence)", "A clause is unfair whenever it states that the applicable law is different from the law of the consumer's place of residence.", "The limitation of liability clause specifies for what actions/events and under what circumstances the providers exclude, limit, or reduce their liability, the duty to compensate damages, and/or includes blanket phrases like 'to the fullest extent permissible by law'. Such a clause is unfair unless it pertains to a force majeure case", "A clause is unfair whenever it states that the provider has the right to suspend and/or terminate the service, the contract, or the consumer's account for any or no reason, with or without notice", "A clause is unfair whenever it states that the consumer is bound by the terms of use/service simply by using the service, downloading the app, or visiting the website", "A clause is unfair if it explicitly state that, simply by using the service, the consumer consents to the processing of personal data as described in the privacy policy, and/or state that the privacy policy is incorporated into and forms part of the terms, especially if it is preceded by a 'contract by using' clause"]

conf = config.update_config({}, 'configs/default.yaml')
model = llm.model_from_config(conf['generation']['model'], disable_tqdm=False)

demos = [10]
results = {(i): {} for i in demos}

for i in demos:

    for label in range(1, 10):

        eval_template = \
        """You are analyzing terms of service. 
        [PROMPT]
        {cat_descr}
        If the clause is unfair respond with 'unfair' otherwise respond with 'fair'.
        Provide only 'unfair' or 'fair' as the output. Do not include any explanations or additional text.

        Input: [INPUT]
        Output: [OUTPUT]"""

        print(f"Running evaluation for num demo: {i} and label: {label}")

        eval_template = eval_template.format(cat_descr=category_definitions[label-1])

        fair_clauses_train = fair_cat_clauses_train[list(fair_cat_clauses_idxs_train.keys())[label-1]]["text"]
        fair_clauses_val = fair_cat_clauses_val[list(fair_cat_clauses_idxs_train.keys())[label-1]]["text"]
        
        unfair_clauses_train = unfair_cat_clauses_train[list(fair_cat_clauses_idxs_val.keys())[label-1]]["text"]
        unfair_clauses_val = unfair_cat_clauses_val[list(fair_cat_clauses_idxs_val.keys())[label-1]]["text"]
        
        clauses_train = []
        clauses_train.extend(fair_clauses_train)
        clauses_train.extend(unfair_clauses_train)

        clauses_val = []
        clauses_val.extend(fair_clauses_val)
        clauses_val.extend(unfair_clauses_val)
        
        fairness_train = ['fair'] * len(fair_clauses_train) + ['unfair'] * len(unfair_clauses_train)
        fairness_val = ['fair'] * len(fair_clauses_val) + ['unfair'] * len(unfair_clauses_val)
        
        result, demo_fn, model, tokenizer = ape.simple_ape_split(
            train_set=(clauses_train, fairness_train),
            val_set=(clauses_val, fairness_val),
            eval_template=eval_template,
            num_demos = i,
            model=model,
            num_samples = 100
        )

        results[(i)][label] = result.__str__()

        with open("results.txt", "w") as f:
            f.write(str(results))
        
        prompts, scores = result.sorted()

        print(f"Running evaluation for num demo: {i}")

        result_test = ape.simple_eval( 
            train_set=(clauses_train, fairness_train),
            val_set=(clauses_val, fairness_val),
            eval_template=eval_template,
            prompts=prompts[:10],
            eval_model=model
        )

        results_test[i] = result_test.__str__()

        with open("results_test.txt", "w") as f:
            f.write(str(results_test))

with open("results.txt", "w") as f:
    f.write(str(results))

with open("results_test.txt", "w") as f:
    f.write(str(results_test))
'''


###### EASE senza istruzione
'''
prompts, prompts_set, best_exemplars, best_exemplar_indices, best_instructions, instruction_log, r_log, exemplars_log, indices_log, best_r, best_init = ape.run_ease(
    prompt_gen_data=(clauses_train, fairness_train),
    labels_train=labels_train,
    eval_data=(clauses_val, fairness_val),
    labels_val=labels_val,
    test_data=(clauses_test, fairness_test),
    labels_test=labels_test,
    n_init=40,
    total_iter=165,
    NN_training_iter = 50
)

save_path = "results_ease.txt"
with open(save_path, "w") as f:
    f.write(f'Best reward: {best_r}\n')
    f.write(f'Best init: {best_init}\n')
    f.write(f'Best exemplars: {best_exemplars}\n')
    f.write(f'Best exemplar indices: {best_exemplar_indices}\n')
    f.write(f'Best instructions: {best_instructions}\n')
    f.write(f'Best instruction: {instruction_log}\n')
    f.write(f'Best r_log: {r_log}\n')
    f.write(f'Best exemplars_log: {exemplars_log}\n')
    f.write(f'Best indices_log: {indices_log}\n')
    f.write(f'Best prompts: {prompts}\n')
    f.write(f'Best prompts_set: {prompts_set}\n')
'''

###### EASE con istruzione per categoria

category_definitions = ["A clause is unfair whenever arbitration is binding and not optional, and/or should take place in a country different from the consumer's place of residence, and/or be based not on law but on other arbitration rules or the arbiter's discretion.", "A clause is unfair when it specifies if and under what conditions the provider can unilaterally change or modify the contract and/or the service", "A clause is unfair if it gives the provider the right to modify, delete, or remove the user's content, including in-app purchases, under specific conditions or at any time, at their full discretion, for any or no reason, with or without notice or the possibility to retrieve the content", "A clause is unfair whenever it states that judicial proceedings take place away (i.e., in a different city or country from the consumer's place of residence)", "A clause is unfair whenever it states that the applicable law is different from the law of the consumer's place of residence.", "The limitation of liability clause specifies for what actions/events and under what circumstances the providers exclude, limit, or reduce their liability, the duty to compensate damages, and/or includes blanket phrases like 'to the fullest extent permissible by law'. Such a clause is unfair unless it pertains to a force majeure case", "A clause is unfair whenever it states that the provider has the right to suspend and/or terminate the service, the contract, or the consumer's account for any or no reason, with or without notice", "A clause is unfair whenever it states that the consumer is bound by the terms of use/service simply by using the service, downloading the app, or visiting the website", "A clause is unfair if it explicitly state that, simply by using the service, the consumer consents to the processing of personal data as described in the privacy policy, and/or state that the privacy policy is incorporated into and forms part of the terms, especially if it is preceded by a 'contract by using' clause"]

results = {}

conf = config.update_config({}, 'configs/default.yaml')
model = llm.model_from_config(conf['generation']['model'], disable_tqdm=False)

for label in range(2, 10):

    eval_template = \
    """You are analyzing terms of service.
    [PROMPT]
    {cat_descr}
    If the clause is unfair respond with 'unfair' otherwise respond with 'fair'.
    Provide only 'unfair' or 'fair' as the output. Do not include any explanations or additional text.

    Examples:
    [full_DEMO]

    Input: [INPUT]
    Output: [OUTPUT]"""

    sentence_embeddings_template = \
    """You are analyzing terms of service.
    [PROMPT]
    {cat_descr}
    If the clause is unfair respond with 'unfair' otherwise respond with 'fair'.
    Provide only 'unfair' or 'fair' as the output. Do not include any explanations or additional text.

    Examples:
    [full_DEMO]

    """

    #prompts = ['Please, tell whether the input clause is unfair.']
    prompts = ['']
    
    eval_template = eval_template.format(cat_descr=category_definitions[label-1])
    sentence_embeddings_template = sentence_embeddings_template.format(cat_descr=category_definitions[label-1])

    fair_clauses_train = fair_cat_clauses_train[list(fair_cat_clauses_idxs_train.keys())[label-1]]["text"]
    fair_clauses_val = fair_cat_clauses_val[list(fair_cat_clauses_idxs_train.keys())[label-1]]["text"]
    
    unfair_clauses_train = unfair_cat_clauses_train[list(fair_cat_clauses_idxs_val.keys())[label-1]]["text"]
    unfair_clauses_val = unfair_cat_clauses_val[list(fair_cat_clauses_idxs_val.keys())[label-1]]["text"]
    
    clauses_train = []
    clauses_train.extend(fair_clauses_train)
    clauses_train.extend(unfair_clauses_train)

    clauses_val = []
    clauses_val.extend(fair_clauses_val)
    clauses_val.extend(unfair_clauses_val)
        
    fairness_train = ['fair'] * len(fair_clauses_train) + ['unfair'] * len(unfair_clauses_train)
    fairness_val = ['fair'] * len(fair_clauses_val) + ['unfair'] * len(unfair_clauses_val)
    
    prompts, prompts_set, best_exemplars, best_exemplar_indices, best_instructions, instruction_log, r_log, exemplars_log, indices_log, best_r, best_init = ape.run_ease(
        prompt_gen_data=(clauses_train, fairness_train),
        eval_data=(clauses_val, fairness_val),
        conf = conf,
        model_forward_api = model,
        n_init=40,
        total_iter=110,
        NN_training_iter = 50,
        prompts = prompts,
        eval_template = eval_template,
        sentence_embeddings_template = sentence_embeddings_template,
        ape=True
    )

    results[label] = f'Best reward: {best_r}\nBest init: {best_init}\nBest exemplar indices: {best_exemplar_indices}\nBest instructions: {best_instructions}\n'

    save_path = "results_ease.txt"
    with open(save_path, "w") as f:
        f.write(str(results))


###### Eval di EASE + APE SEP
'''
conf = config.update_config({}, 'configs/default.yaml')
model = llm.model_from_config(conf['generation']['model'], disable_tqdm=False)

instructions = [{"a": 'identify which of the input-output pairs indicate fair or unfair conditions for an arbitration.', "ch": 'determine whether the following sentences are considered fair or unfair.', "cr": 'take the first word of a sentence and produce the ethical consideration of the sentence.', "j": 'identify the degree of fairness/unfairness of various jurisdiction and arbitration clauses.', "law": 'give the input: SPECIALNUMBER  legal jurisdiction of which country.', "ltd": 'make two inputs, and to make the second output to match the first output.', "ter": 'make the output fair or unfair depending on the likelihood of the action being taken.', "use": 'write the sentence with all instances of "SPECIALNUMBER" removed.', "pinc": 'give an example of a sentence that is unfair and a sentence that is fair.'}]
ease_idxs = [{"a": [244, 276, 281, 265, 310, 272, 314, 305], "ch": [ 28, 279,  22, 160, 103,  66,  98, 190], "cr": [477, 409, 373, 431, 366, 405, 489, 354], "j": [105, 107,  84,  93, 167, 185, 137, 123], "law": [122, 204, 197, 201, 202, 175, 164, 140], "ltd": [498, 551, 823, 501, 424, 455, 797, 904], "ter": [665, 470, 444, 692, 650, 431, 372, 644], "use": [186, 330, 226, 322, 273, 323, 176, 258], "pinc": [74, 48, 53, 55, 42, 62, 94, 77]}]

category_definitions = ["A clause is unfair whenever arbitration is binding and not optional, and/or should take place in a country different from the consumer's place of residence, and/or be based not on law but on other arbitration rules or the arbiter's discretion.", "A clause is unfair when it specifies if and under what conditions the provider can unilaterally change or modify the contract and/or the service", "A clause is unfair if it gives the provider the right to modify, delete, or remove the user's content, including in-app purchases, under specific conditions or at any time, at their full discretion, for any or no reason, with or without notice or the possibility to retrieve the content", "A clause is unfair whenever it states that judicial proceedings take place away (i.e., in a different city or country from the consumer's place of residence)", "A clause is unfair whenever it states that the applicable law is different from the law of the consumer's place of residence.", "The limitation of liability clause specifies for what actions/events and under what circumstances the providers exclude, limit, or reduce their liability, the duty to compensate damages, and/or includes blanket phrases like 'to the fullest extent permissible by law'. Such a clause is unfair unless it pertains to a force majeure case", "A clause is unfair whenever it states that the provider has the right to suspend and/or terminate the service, the contract, or the consumer's account for any or no reason, with or without notice", "A clause is unfair whenever it states that the consumer is bound by the terms of use/service simply by using the service, downloading the app, or visiting the website", "A clause is unfair if it explicitly state that, simply by using the service, the consumer consents to the processing of personal data as described in the privacy policy, and/or state that the privacy policy is incorporated into and forms part of the terms, especially if it is preceded by a 'contract by using' clause"]

eval_template = \
"""You are analyzing terms of service.
[PROMPT]
{cat_descr}
If the clause is unfair respond with 'unfair' otherwise respond with 'fair'.
Provide only 'unfair' or 'fair' as the output. Do not include any explanations or additional text.

Examples:
[full_DEMO]

Input: [INPUT]
Output: [OUTPUT]"""


results = {}

for label in range(1, 10):
    
    eval_template = eval_template.format(cat_descr=category_definitions[label-1])

    fair_clauses_train = fair_cat_clauses_train[list(fair_cat_clauses_idxs_train.keys())[label-1]]["text"]
    fair_clauses_val = fair_cat_clauses_val[list(fair_cat_clauses_idxs_train.keys())[label-1]]["text"]
        
    unfair_clauses_train = unfair_cat_clauses_train[list(fair_cat_clauses_idxs_val.keys())[label-1]]["text"]
    unfair_clauses_val = unfair_cat_clauses_val[list(fair_cat_clauses_idxs_val.keys())[label-1]]["text"]
        
    clauses_train = []
    clauses_train.extend(fair_clauses_train)
    clauses_train.extend(unfair_clauses_train)

    clauses_val = []
    clauses_val.extend(fair_clauses_val)
    clauses_val.extend(unfair_clauses_val)
        
    fairness_train = ['fair'] * len(fair_clauses_train) + ['unfair'] * len(unfair_clauses_train)
    fairness_val = ['fair'] * len(fair_clauses_val) + ['unfair'] * len(unfair_clauses_val)
    
    train_set = (clauses_train, fairness_train)

    demo_data = ([train_set[0][j] for j in ease_idxs[0][list(fair_cat_clauses_idxs_train.keys())[label-1]]], [train_set[1][j] for j in ease_idxs[0][list(fair_cat_clauses_idxs_train.keys())[label-1]]])

    result_manual = ape.simple_eval( 
        train_set=train_set,
        val_set=(clauses_val, fairness_val),
        eval_template=eval_template,
        prompts=[instructions[0][list(fair_cat_clauses_idxs_train.keys())[label-1]]],
        demo_data=demo_data,
        eval_model=model)

    results[label] = result_manual

    with open("results_manual.txt", "w") as f:
        for few_shot_idx, result_manual in results.items():
            f.write(f"Label: {label}\n")
            f.write(f"Result: {result_manual}\n\n")
'''


###### EASE + APE
'''
prompts, prompts_set, best_exemplars, best_exemplar_indices, best_instructions, instruction_log, r_log, exemplars_log, indices_log, best_r, best_init = ape.run_ease(
    prompt_gen_data=(clauses_train, fairness_train),
    labels_train=labels_train,
    eval_data=(clauses_val, fairness_val),
    labels_val=labels_val,
    test_data=(clauses_test, fairness_test),
    labels_test=labels_test,
    n_init=40,
    total_iter=165,
    NN_training_iter = 50,
    ape = True, ###cambiare num_demos in default.yaml
)

save_path = "results_ease.txt"
with open(save_path, "w") as f:
    f.write(f'Best reward: {best_r}\n')
    f.write(f'Best init: {best_init}\n')
    f.write(f'Best exemplars: {best_exemplars}\n')
    f.write(f'Best exemplar indices: {best_exemplar_indices}\n')
    f.write(f'Best instructions: {best_instructions}\n')
    f.write(f'Best instruction: {instruction_log}\n')
    f.write(f'Best r_log: {r_log}\n')
    f.write(f'Best exemplars_log: {exemplars_log}\n')
    f.write(f'Best indices_log: {indices_log}\n')
    f.write(f'Best prompts: {prompts}\n')
    f.write(f'Best prompts_set: {prompts_set}\n')
'''

###### Iterative APE
'''
instructions, scores = result.sorted()
print(result)
res = ape.iterative_ape(eval_template,
                  (clauses_val, fairness_val), 
                  labels_val,
                  (clauses_train, fairness_train),
                  labels_train,
                  instructions,
                  model,
                  tokenizer)
print(res)
'''

###### Eval 504 di EASE e EASE + APE
'''
ease_idxs = []
manual_prompts = []

print(f'len prompt {len(manual_prompts)}')
print(f'len idxs {len(ease_idxs)}')

conf = config.update_config({}, 'configs/default.yaml')
model = llm.model_from_config(conf['evaluation']['model'])
train_set = (clauses_train, fairness_train)

print(f'len prompts {len(manual_prompts)}')
print(f'len idxs {len(ease_idxs)}')

results = {}

for i, few_shot_idx in enumerate(ease_idxs):
    demo_data = ([train_set[0][j] for j in few_shot_idx], [train_set[1][j] for j in few_shot_idx])

    if manual_prompts[i] == '':
        eval_template = '[full_DEMO]\n\nInput: [INPUT]\nOutput: [OUTPUT]'
    else:
        eval_template = \
        """Instruction: [PROMPT]

        [full_DEMO]

        Input: [INPUT]
        Output: [OUTPUT]"""

    print(f"Running evaluation for few-shot set {i}: {few_shot_idx}")
    result_manual = ape.simple_eval( 
        train_set=train_set,
        labels_train=labels_train,
        val_set=(clauses_val, fairness_val),
        labels_val=labels_val,
        eval_template=eval_template,
        prompts=[manual_prompts[i]],
        num_samples=252,
        demo_data=demo_data,
        eval_model=model)

    results[tuple(few_shot_idx)] = result_manual

    with open("results_manual.txt", "w") as f:
        for few_shot_idx, result_manual in results.items():
            f.write(f"Few-shot indices: {few_shot_idx}\n")
            f.write(f"Result: {result_manual}\n\n")

#prompts, scores = result_manual.sorted()
#print(''.join(f'{scores[i]:.2f}: {prompts[i]}\n' for i in range(len(prompts))))
'''

###### Class count nei few shot trovati
'''
ease_idxs = []

class_ids = {}
class_count = {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0}

for idxs in ease_idxs:
    ids = []
    
    id = ds_train["labels"][idxs]
    ids.append(id)
    
    for i in id:
        class_count[i] += 1

    class_ids[idxs] = ids

print(class_ids)
print(class_count)
'''
'''
###### Class count in tutto il dataset
class_count_binary = 0
class_count = {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0}
for idx in range(len(ds_train)):
    id = ds_train["labels"][idx]
    if 0 not in id:
        class_count_binary += 1
    for i in id:
        class_count[i] += 1
for idx in range(len(ds_val)):
    id = ds_val["labels"][idx]
    if 0 not in id:
        class_count_binary += 1
    for i in id:
        class_count[i] += 1
#for i in range(1, 10):
#    class_count[i] = class_count[i] + len(fair_cat_clauses_idxs_val[list(fair_cat_clauses_idxs_val.keys())[i-1]])
for idx in range(len(ds_test)):
    id = ds_test["labels"][idx]
    if 0 not in id:
        class_count_binary += 1
    for i in id:
        class_count[i] += 1
print(class_count)
print(class_count_binary)
fair = []

for i in range(1, 10):
    fair.extend(fair_cat_clauses_idxs_val[list(fair_cat_clauses_idxs_val.keys())[i-1]])
    fair.extend(fair_cat_clauses_idxs_train[list(fair_cat_clauses_idxs_val.keys())[i-1]])

#create a set of fair indices to remove duplicates
fair_set = set(fair)
print(f'fair_set {len(fair_set)}')


class_count = {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0}
for idx in range(len(ds_val)):
    id = ds_train["labels"][idx]
    text = ds_train["text"][idx]
    for i in id:
        if class_count[i] == 0:
            class_count[i] += len(text)
        else:
            class_count[i] += len(text)
            class_count[i] = class_count[i]/2


print(class_count)
print(len(ds_val))
'''