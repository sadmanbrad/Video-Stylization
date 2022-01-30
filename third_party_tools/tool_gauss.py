import os

file_naming_format = "%03d"  # name of input files, e.g., %03d if files are named 001.png, 002.png
# path to the forward flow files (computed by third_party_tools/disflow)
flow_fwd_files = "flow_fwd" + "/" + file_naming_format + ".A2V2f"
# path to the backward flow files (computed by third_party_tools/disflow)
flow_bwd_files = "flow_bwd" + "/" + file_naming_format + ".A2V2f"

keyframes_path = 'keyframes'
gdisko_gauss_r10_s10_dir = "input_gdisko_gauss_r10_s10"  # path to the result gauss r10 s10 sequence
gdisko_gauss_r10_s15_dir = "input_gdisko_gauss_r10_s15"  # path to the result gauss r10 s15 sequence

keyframe_files = keyframes_path + "/" + file_naming_format + '.png'
gdisko_gauss_r10_s10_files = gdisko_gauss_r10_s10_dir + "/" + file_naming_format + ".png"
gdisko_gauss_r10_s15_files = gdisko_gauss_r10_s15_dir + "/" + file_naming_format + ".png"


def generate_aux_video(start_index, end_index):
    if not os.path.exists(gdisko_gauss_r10_s10_dir):
        os.mkdir(gdisko_gauss_r10_s10_dir)

    if not os.path.exists(gdisko_gauss_r10_s15_dir):
        os.mkdir(gdisko_gauss_r10_s15_dir)

    keyframes_str = ""
    keyframes_list_dir = os.listdir(keyframes_path)
    for keyframe in keyframes_list_dir:
        keyframes_str += keyframe.replace(".png", "").replace(".jpg", "")
        keyframes_str += " "

    if os.system(f".\\third_party_tools\\gauss\\gauss.exe {keyframe_files} {flow_fwd_files} {flow_bwd_files} {start_index} {end_index} "
                 f"{len(keyframes_list_dir)} {keyframes_str} 10 10 {gdisko_gauss_r10_s10_files}") != 1:
        raise Exception('Failed to generate auxiliary channels')
    # if os.system(f"gauss {keyframe_files} {flow_fwd_files} {flow_bwd_files} {start_index} {end_index} "
    #              f"{len(keyframes_list_dir)} {keyframes_str} 10 15 {gdisko_gauss_r10_s15_files}") != 1:
    #     raise 'Failed to generate auxiliary channels'
