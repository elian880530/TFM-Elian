# Pythono3 code to rename multiple
# files in a directory or folder

# importing os module
import os

''' 
'''
# Function to rename multiple files
def mainA():
    i = 0
    var = 0
    letra = "a "

    for filename in os.listdir("G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/a"):
        dst = letra + str(var) + "-" + "Parado-" + str(i) + ".png"
        src = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/a/' + filename
        dst = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/a/' + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)
        i += 1
        var += 1



'''  
'''
def mainB():
    i = 0
    var = 500
    letra = "b "

    for filename in os.listdir("G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/b"):
        dst = letra + str(var) + "-" + "Agachado-" + str(i) + ".png"
        src = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/b/' + filename
        dst = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/b/' + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)
        i += 1
        var += 1


'''
'''
def mainC():
    i = 0
    var = 1000
    letra = "c "
    
    for filename in os.listdir("G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/c"):
        dst = letra + str(var) + "-" + "CaidaTipoFin-" + str(i) + ".png"
        src = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/c/' + filename
        dst = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/c/' + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)
        i += 1
        var += 1


''' 
'''
def mainD():
    i = 0
    var = 1500
    letra = "d "

    for filename in os.listdir("G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/d"):
        dst = letra + str(var) + "-" + "Agachado-" + str(i) + ".png"
        src = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/d/' + filename
        dst = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/d/' + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)
        i += 1
        var += 1


''' 
'''
def mainE():
    i = 0
    var = 2000
    letra = "e "

    for filename in os.listdir("G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/e"):
        dst = letra + str(var) + "-" + "Parado-" + str(i) + ".png"
        src = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/e/' + filename
        dst = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/e/' + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)
        i += 1
        var += 1



''' 
'''
def mainF():
    i = 0
    var = 2500
    letra = "f "

    for filename in os.listdir("G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/f"):
        dst = letra + str(var) + "-" + "Vacio-" + str(i) + ".png"
        src = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/f/' + filename
        dst = 'G:/doc tesis/cod proyecto/Ouch/BD Imagenes and Videos/Casa-Sa-Pobla-CaidasReales/f/' + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)
        i += 1
        var += 1



# Driver Code
if __name__ == '__main__':
    # Calling main() function
    mainA()
    mainB()
    mainC()
    mainD()
    mainE()
    mainF()