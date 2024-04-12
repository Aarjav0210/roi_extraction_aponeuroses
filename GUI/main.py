import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils.denoiser import denoise_image
from utils.enhancer import isolate_intensity
from utils.detector import edge_detection, hough_transform
from utils.extractor import get_quadrilateral_coordinates

def load_image(file_path):
    """
    Load an image from a file and return it as a numpy array.
    """
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # crop image to 512x512
    if image.shape[0] > 512 and image.shape[1] > 512:
        image = image[:512, :512]
    if image.shape[0] < 512 or image.shape[1] < 512:
        image = cv2.resize(image, (512, 512))

    return np.asarray(image)

def process_image(image_path, technique, method, flip):
    """
    Process the image based on the selected method and return the processed image and text.
    """
    image = load_image(image_path)
    if flip:
        image = cv2.flip(image, 1)
    denoised_image = denoise_image(image, technique)  # Assuming this function returns a numpy array
    enhanced_image = isolate_intensity(denoised_image, method)
    edges = edge_detection(enhanced_image)
    _, line_d, line_s = hough_transform(edges, visualise=denoised_image)
    # # Check if output is the same as edges
    # if edges is output:
    #     output = denoised_image

    max_value = np.max(denoised_image)
    min_value = np.min(denoised_image)
    # Normalise the image to 0-255
    if max_value < 1:
        # Normalise according to the min and max values
        denoised_image = (denoised_image - min_value) / (max_value - min_value) * 255

    coordinates = get_quadrilateral_coordinates(line_d, line_s, edges.shape[1], image.shape[1])

    draw_coordinates = coordinates.copy()
    if len(coordinates) == 0:
        draw_coordinates = [[0, 0], [0, image.shape[0]-1], [image.shape[1]-1, image.shape[0]-1], [image.shape[1]-1, 0]]
        coordinates = "No ROI found. Assume the whole image to be the ROI."
    # Draw the quadrilateral on the image
    # Make the image RGB
    if len(denoised_image.shape) == 2:
        denoised_image = cv2.cvtColor(denoised_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if len(draw_coordinates) > 0:
        thickness = 3 if denoised_image.shape[0] >= 512 else 1
        if technique == 'Average Pooling':
            # Correctly scale the coordinates d
            scale_x = denoised_image.shape[1] / image.shape[1]
            scale_y = denoised_image.shape[0] / image.shape[0]
            draw_coordinates = [[int(x * scale_x), int(y * scale_y)] for x, y in draw_coordinates]
            print(draw_coordinates)


        cv2.polylines(denoised_image, [np.array(draw_coordinates)], isClosed=True, color=(255, 0, 0), thickness=thickness)
    # if len(coordinates) == 0:
    #     return denoised_image, "No ROI found. Assume the whole image to be the ROI."
    return denoised_image, f"Coordinates: {coordinates}"

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Image Processor")
        self.root.geometry('1500x800')

        # Label for the "Open Image" button
        ttk.Label(self.root, text="Select Image:").grid(row=0, column=0, sticky='e', padx=5, pady=5)

        # Open Image Button
        self.file_button = ttk.Button(self.root, text="Open Image", command=self.open_file, bootstyle=PRIMARY)
        self.file_button.grid(row=0, column=1, padx=10, pady=10)

        # Label for Techniques Combobox
        ttk.Label(self.root, text="Select Technique:").grid(row=1, column=0, sticky='e', padx=5, pady=5)

        # Techniques Combobox
        self.techniques = ['NLM', 'Bilateral', 'Gaussian', 'Average Pooling', 'Noise2Noise']
        self.technique_var = tk.StringVar()
        self.technique_dropdown = ttk.Combobox(self.root, textvariable=self.technique_var, values=self.techniques, bootstyle=PRIMARY)
        self.technique_dropdown.grid(row=1, column=1, padx=10, pady=10)
        self.technique_dropdown.current(0)

        # Label for Methods Combobox
        ttk.Label(self.root, text="Select Method:").grid(row=2, column=0, sticky='e', padx=5, pady=5)

        # Methods Combobox
        self.methods = ['Q1', 'AMI']
        self.method_var = tk.StringVar()
        self.method_dropdown = ttk.Combobox(self.root, textvariable=self.method_var, values=self.methods, bootstyle=PRIMARY)
        self.method_dropdown.grid(row=2, column=1, padx=10, pady=10)
        self.method_dropdown.current(0)

        # Label for Flip Image Checkbutton
        ttk.Label(self.root, text="Image Options:").grid(row=3, column=0, sticky='e', padx=5, pady=5)

        # Flip Image Checkbutton
        self.flip_var = tk.IntVar()
        self.flip_checkbox = ttk.Checkbutton(self.root, text="Flip Image", variable=self.flip_var, bootstyle=PRIMARY)
        self.flip_checkbox.grid(row=3, column=1, padx=10, pady=10)

        # Process Button
        self.process_button = ttk.Button(self.root, text="Process", command=self.process_image, bootstyle=PRIMARY)
        self.process_button.grid(row=4, column=0, columnspan=2, pady=10)

        self.input_image_frame = tk.Frame(self.root)
        self.input_image_frame.grid(row=0, column=2, rowspan=4)
        self.output_image_frame = tk.Frame(self.root)
        self.output_image_frame.grid(row=0, column=3, rowspan=4)

        self.output_text = tk.Label(self.root, text="")
        self.output_text.grid(row=5, column=2, columnspan=2)

    def open_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            self.display_image(file_path, self.input_image_frame)

    def display_image(self, image_path, canvas_frame):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        fig = Figure(figsize=(5, 4), dpi=100)
        plot = fig.add_subplot(111)
        plot.imshow(img, cmap='gray')
        plot.axis('off')
        for widget in canvas_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def process_image(self):
        selected_technique = self.technique_var.get()
        selected_method = self.method_var.get()
        flip_image = self.flip_var.get()
        processed_image, output_text = process_image(self.image_path, selected_technique, selected_method, flip_image)
        self.display_array_as_image(processed_image, self.output_image_frame)
        self.output_text.config(text=output_text)

    def display_array_as_image(self, array, canvas_frame):
        fig = Figure(figsize=(5, 4), dpi=100)
        plot = fig.add_subplot(111)
        plot.imshow(array, cmap='gray')
        plot.axis('off')
        for widget in canvas_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()




# import tkinter as tk
# from tkinter import ttk, filedialog
# import numpy as np
# import cv2
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# # Assuming denoise_image is your custom function for image processing
# from utils.denoiser import denoise_image
# from utils.enhancer import isolate_intensity
# from utils.detector import edge_detection, hough_transform
# from utils.extractor import get_quadrilateral_coordinates

# def load_image(file_path):
#     """
#     Load an image from a file and return it as a numpy array.
#     """
#     image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#     return np.asarray(image)

# def process_image(image_path, technique, method, flip):
#     """
#     Process the image based on the selected method and return the processed image and text.
#     """
#     image = load_image(image_path)
#     if flip:
#         image = cv2.flip(image, 1)
#     denoised_image = denoise_image(image, technique)  # Assuming this function returns a numpy array
#     enhanced_image = isolate_intensity(denoised_image, method)
#     edges = edge_detection(enhanced_image)
#     output, line_d, line_s = hough_transform(edges, visualise=image)

#     coordinates = get_quadrilateral_coordinates(line_d, line_s, edges.shape[1], image.shape[0])
#     return output, f"Coordinates: {coordinates}"

# class ImageProcessorApp:
#     def __init__(self, root):
#         self.root = root
#         self.setup_ui()

#     def setup_ui(self):
#         self.root.title("Image Processor")
#         self.root.geometry('1500x800')
#         self.root.resizable(False, False)

#         self.file_button = tk.Button(self.root, text="Open Image", command=self.open_file)
#         self.file_button.grid(row=0, column=0, padx=10, pady=10)

#         self.techniques = ['NLM', 'Bilateral', 'Gaussian', 'Average Pooling', 'Noise2Noise']
#         self.technique_var = tk.StringVar()
#         self.technique_dropdown = ttk.Combobox(self.root, textvariable=self.technique_var, values=self.techniques)
#         self.technique_dropdown.grid(row=1, column=0, padx=10, pady=10)
#         self.technique_dropdown.current(0)

#         self.methods = ['Q1', 'AMI']
#         self.method_var = tk.StringVar()
#         self.method_dropdown = ttk.Combobox(self.root, textvariable=self.method_var, values=self.methods)
#         self.method_dropdown.grid(row=2, column=0, padx=10, pady=10)
#         self.method_dropdown.current(0)

#         # Flip checkbox
#         self.flip_var = tk.IntVar()
#         self.flip_checkbox = tk.Checkbutton(self.root, text="Flip Image", variable=self.flip_var)
#         self.flip_checkbox.grid(row=3, column=0, padx=10, pady=10)


#         self.process_button = tk.Button(self.root, text="Process", command=self.process_image)
#         self.process_button.grid(row=4, column=0, padx=10, pady=10)

#         # Create frames for matplotlib figures
#         self.input_image_frame = tk.Frame(self.root)
#         self.input_image_frame.grid(row=0, column=1, rowspan=4)
#         self.output_image_frame = tk.Frame(self.root)
#         self.output_image_frame.grid(row=0, column=2, rowspan=4)

#         self.output_text = tk.Label(self.root, text="")
#         self.output_text.grid(row=5, column=2, columnspan=2)

#     def open_file(self):
#         file_path = filedialog.askopenfilename()
#         if file_path:
#             self.image_path = file_path
#             self.display_image(file_path, self.input_image_frame)

#     def display_image(self, image_path, canvas_frame):
#         img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         fig = Figure(figsize=(5, 4), dpi=100)
#         plot = fig.add_subplot(111)
#         plot.imshow(img, cmap='gray')
#         plot.axis('off')
#         canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
#         canvas.draw()
#         canvas.get_tk_widget().pack()

#     def process_image(self):
#         selected_technique = self.technique_var.get()
#         selected_method = self.method_var.get()
#         flip_image = self.flip_var.get()
#         processed_image, output_text = process_image(self.image_path, selected_technique, selected_method, flip_image)
#         self.display_array_as_image(processed_image, self.output_image_frame)
#         self.output_text.config(text=output_text)

#     def display_array_as_image(self, array, canvas_frame):
#         fig = Figure(figsize=(5, 4), dpi=100)
#         plot = fig.add_subplot(111)
#         plot.imshow(array, cmap='gray')
#         plot.axis('off')
#         for widget in canvas_frame.winfo_children():
#             widget.destroy()
#         canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
#         canvas.draw()
#         canvas.get_tk_widget().pack()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ImageProcessorApp(root)
#     root.mainloop()
