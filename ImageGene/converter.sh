ffmpeg -r 60 -f image2 -i {picture_dir}/%d0.png -vcodec libx264 -crf 15 -vf scale={width}:{height}:flags=neighbor -pix_fmt gray {filename}.mp4
