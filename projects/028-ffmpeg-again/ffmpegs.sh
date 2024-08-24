# file conversion
ffmpeg -i input.mov output.mp4

# audio extraction
ffmpeg -i input.mp4 output.mp3

# audio/video combining
ffmpeg -i vid.mp4 -i audio.mp3 -c:v copy -c:a aac output.mp4

# resize video
ffmpeg -i vid.mp4 -vf "scale=120:20" out.mp4

# trim 
ffmpeg -i vid.mp4 -ss 00:00:00 -to 00:00:02 -c copy output.mp4

# create slideshow
ffmpeg -framerate 1 -i img%d.png -c:v libx264 -r 30 -pix_fmt yuv420p slideshow.mp4

# extract frames
ffmpeg -i video.mp4 -vf "fps=1" frame%d.png

# subtitles
ffmpeg -i vid.mp4 -i subtitles.srt -c copy -c:s mov_text output.mp4

# video to gif
ffmpeg -i vid.mp4 -vf "fps=10,scale=320:-1,flags=lanczos" -c:v gif -loop 0 vid.gif

# extract thumbnail
ffmpeg -i vid.mp4 -ss 00:00:10 -vframes 1 output.jpeg

# speed up video
ffmpeg -i vid.mp4 -filter:v "pts=0.5*PTS" output.mp4
# note -vf = -filter:v

# normalize audio
ffmpeg -i input.mp4 -af "loudnorm" output.mp4

# convert for streaming
ffmpeg -i input.mp4 -code: copy -start_number 0 -hls_time 10 -hls_list_size 0 -f hls index.m3u8

# rotate 90 deg
ffmpeg -i input.mp4 -vf "transpose=1" output.mp4

# overlay an image
ffmpeg -i input.mp4 -i image.jpg -filter_complex "overlay=10:10" output.mp4

# split video
ffmpeg -i input.mp4 -c copy -map 0 -segment_time 00:10:00 -f segment output%03d.mp4

# stabilize
ffmpeg -i input.mp4 -vf deshake output.mp4

# merge
ffmpeg -f concat -safe 0 -i filelist.txt -c copy output.mp4

# loop
ffmpeg -stream_loop 5 -i input.mp4 -c copy output.mp4

# capture from webcam
ffmpeg -f v412 -framerate 25 -video_size 640x640 -i /dev/video0 output.mkv
