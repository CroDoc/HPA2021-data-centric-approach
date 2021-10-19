python segmentator.py -i CAM_images -n CAM_nuclei -c CAM_cells -b 128
python image_cutter.py -i CAM_images -m CAM_cells -o CAM_dataset
