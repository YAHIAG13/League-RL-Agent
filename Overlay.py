import tkinter as tk
import pyautogui

class Overlay(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.geometry("200x100+100+100")  # Set the size and position of the overlay window
        self.title("Overlay Window")

        # Add your UI elements here
        self.label = tk.Label(self, text="Overlay Content")
        self.label.pack()

def get_app_window_position():
    # Use PyAutoGUI to get the position of the window of the application or game
    # Here, I'm just assuming the window title is "Your Application Window Title"
    window_pos = pyautogui.locateOnScreen("assets/play_btn.pnefbg")  # You can also use a screenshot of the window title
    return window_pos[0], window_pos[1]  # Return the x and y coordinates of the top-left corner

def main():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Create the overlay window
    overlay = Overlay(root)

    # Get the position of the application or game window
    app_x, app_y = get_app_window_position()

    # Position the overlay window on top of the application or game window
    overlay.geometry("+{}+{}".format(app_x, app_y))

    # Run the Tkinter event loop
    overlay.mainloop()

if __name__ == "__main__":
    main()
