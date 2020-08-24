import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import os

def main():
    if not os.path.exists(args.directory):
        os.mkdir(args.directory)
    try:
        config = rs.config()

        # En el argumento que se inicializa cuando se ejecuta el main se pasa la dirección y archivo bag.
        rs.config.enable_device_from_file(config, args.input)

        # Se llama a la función colorer para darle el formato deseado a las imágenes
        colorizer = rs.colorizer()

        # Se escogió el esquema 2 que en el visor es el WhiteToBlack
        colorizer.set_option(rs.option.color_scheme, 2)

        # Se llama al objeto rs con la configuración de la cámara: profundidad,tamaño,formato y cant frames
        pipeline = rs.pipeline()
        #config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
        config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 60)

        # Lanzo el servicio
        pipeline.start(config)
        i = 0

        while True:
            print("Saving frame:", i)
            frames = pipeline.wait_for_frames()

            # Check if new frame is ready
            for f in frames:

                # Obtengo el frame de profundidad
                depth_frame = frames.get_depth_frame()

                # Le doy color aplicando el tema WhiteToBlack
                colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

                # Lo guardo en la dirección que se desea con formato png
                #cv2.imwrite(args.directory + "/Casa-Pla-Parado/Parado-" + str(i + 6) + ".png", colorized_depth)
                cv2.imwrite(args.directory + "/Casa-Sa-Pobla-CaidasReales/CaidasReales-" + str(i + 5000) + ".png", colorized_depth)

                # Continúo con el proceso de iteración
                i += 1

            # wait logic until frame is ready
            else:
                print("Waiting for frame to be ready")

    finally:
        pass





if __name__ == "__main__":

    # Se crean las variables argumentales que posteriormente tomarán valores
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="Path to save the images")
    parser.add_argument("-i", "--input", type=str, help="Bag file to read")

    # Se inicializan los argumentos directory e input con los valores deseados
    args = parser.parse_args(["--directory","C:/Users/EGH/PycharmProjects/Ouch","--input","20200714_123831.bag"])

    # Llamada al método main
    main()
