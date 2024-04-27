# convert format
ffmpeg -i input.mp4 output.mov

# extract audio
ffmpeg -i input.mp4 output.mp3

# combine video and audio
ffmpeg -i video.mp4 -i audio.mp3 -c:v copy -c:a aac output.mp4

# resize video
ffmpeg -i video.mp4 -vf "scale=1280:720" output.mp4

# trim video
ffmpeg -i input.mp4 -ss 00:00:10 -to 00:01:00 -c copy output.mp4

# create slideshow form images
ffmpeg -framerate 1 -i img%d.png -c:v libx264 -r 30 -pix_fmt yuv420p slideshow.mp4

# extract frames
ffmpeg -i video.mp4 -vf "fps=1" frame%d.png

# add subtitles
ffmpeg -i input.mp4 -i subtitles.srt -c copy -c:s mov_text output.mp4
