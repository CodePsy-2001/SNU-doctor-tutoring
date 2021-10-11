from pytube import YouTube

yt = YouTube("https://www.youtube.com/watch?v=JyECrGp-Sw8")

myvideo = yt.streams.get_by_itag(22)

myvideo.download()
