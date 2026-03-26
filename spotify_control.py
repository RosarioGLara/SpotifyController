import spotipy
from spotipy.oauth2 import SpotifyOAuth

class SpotifyController:
     def __init__(self):     
          self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
          client_id="ADD YOUR API KEY HERE",
          client_secret="ADD YOUR API KEY HERE",
          redirect_uri="http://127.0.0.1:8888/callback",
          scope="user-read-playback-state,user-modify-playback-state"
          ))

     def get_current(self):
          return self.sp.current_playback()
     
     def play(self):
          current = self.sp.current_playback()

          if not current or not current['is_playing']:
               self.sp.start_playback()

     def pause(self):
          current = self.sp.current_playback()

          if current and current['is_playing']:
               self.sp.pause_playback()
          else:
               print("Nothing to pause")

     def next_song(self):
          self.sp.next_track()

     def volume_up(self, step=10):
          current = self.sp.current_playback()

          if current and current.get('device'):
               current_volume = current['device']['volume_percent']
               new_volume = min(100, current_volume + step)

               self.sp.volume(new_volume)
               print(f"Volume: {new_volume}")

     def volume_down(self):
          self.sp.volume(30)

     def toggle_play(self):
          current = self.get_current()

          if current and current['is_playing']:
               self.pause()
          else:
               self.play()
     def devices(self):
          print(self.sp.devices())
