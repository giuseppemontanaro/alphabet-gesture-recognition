import threading
import keyboard

shortcut = 'Cmd + S'

def performShortcut():
  threading.Timer(150.0, performShortcut).start() #<-- set timer in second
  keyboard.press_and_release(shortcut)

performShortcut()