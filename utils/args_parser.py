import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    #Argument for hyperspectral dataset (Avon, RIT Campus)
    #Avon ('avon') comes from the SHARE2012 datacollect
    #RIT Campus ('ritcampus') comes from the SHARE2010 datacollect
    parser.add_argument('-hsidataset', type=str, default='avon',
                        choices=['avon', 'ritcampus'])
    
    #Specify target spectral signature
    #There are two felt targets contained on the scene: red felt and blue felt
    parser.add_argument('-targetsig', type=str, default='red', choices=['red', 'blue'])
    
    #Specify root directory where images live
    parser.add_argument('-root', type=str, default='./images')
    
    args = parser.parse_args()
    return args
 