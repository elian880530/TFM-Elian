# Improting Image class from PIL module
from PIL import Image
import os.path

#path = "C:/Users/EGH/PycharmProjects/Ouch/Casa-Pla-Agachado"
#path = "C:/Users/EGH/PycharmProjects/Ouch/clean-noise"
#path = "C:/Users/EGH/PycharmProjects/Ouch/Casa-Sa-Pobla-Agachado-Clean"
#path = "C:/Users/EGH/PycharmProjects/Ouch/Casa-Sa-Pobla-CaidaTipoFin-Clean"
#path = "C:/Users/EGH/PycharmProjects/Ouch/Casa-Sa-Pobla-Parado-Clean"
#path = "C:/Users/EGH/PycharmProjects/Ouch/Casa-Sa-Pobla-Vacio-Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/1_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/2_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/3_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/4_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/5_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/6_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/7_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/8_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/9_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/10_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/11_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/12_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/13_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/14_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/15_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/16_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/17_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/18_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/19_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/20_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/21_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/22_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/23_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/24_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/25_Caida_Clean"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-Agachado-Agachado"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-Agachado-CaidaTipoFin"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-Parado-Agachado"
#path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-Parado-CaidaTipoFin"
path = "G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-Parado-Parado"

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