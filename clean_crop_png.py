# Improting Image class from PIL module
from PIL import Image
import os.path

#path = "C:/Users/EGH/PycharmProjects/Ouch/Casa-Pla-Agachado"
#path = "C:/Users/EGH/PycharmProjects/Ouch/clean-noise"
#path = "C:/Users/EGH/PycharmProjects/Ouch/Casa-Sa-Pobla-Agachado-Clean"
#path = "C:/Users/EGH/PycharmProjects/Ouch/Casa-Sa-Pobla-CaidaTipoFin-Clean"
#path = "C:/Users/EGH/PycharmProjects/Ouch/Casa-Sa-Pobla-Parado-Clean"
path = "C:/Users/EGH/PycharmProjects/Ouch/Casa-Sa-Pobla-Vacio-Clean"

#IMG_SIZE_Height = 710
#IMG_SIZE_Width = 860

#IMG_SIZE_Height = 270
#IMG_SIZE_Width = 300

#IMG_SIZE_Height = 270
#IMG_SIZE_Width = 480

dirs = os.listdir(path)

# Function to recort multiple files
def main():
    i = 0
    for item in dirs:
        fullpath = os.path.join(path, item)  # corrected
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            #Tipo de retorno: Imagen (Devuelve una región rectangular como (izquierda, superior, derecha, inferior) -tupla).
            #imCrop = im.crop((0, 0, 860, 710))  # corrected
            imCrop = im.crop((70, 0, 370, 270))   # corrected
            imCrop.save(f + ".png", "PNG", quality=100)
            print(i)
            i += 1


# Calling main() function
if __name__ == '__main__':
    main()