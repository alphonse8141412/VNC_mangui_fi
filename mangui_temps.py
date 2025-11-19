#!/usr/bin/env python3
"""
TEST AVEC ALPHONSE SEULEMENT - VERSION CORRIGÉE
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

print("🎯 MANGUI FI - TEST ALPHONSE")
print("=" * 40)

class TestAlphonse:
    def __init__(self):
        self.visage_alphonse = None
        self.pointages = []
        
        print("📁 Chargement de la photo d'Alphonse...")
        self.charger_alphonse()
    
    def charger_alphonse(self):
        """Charge la photo d'Alphonse comme référence"""
        # Chercher le fichier Alphonse
        for f in os.listdir("dev_data"):
            if "alphonse" in f.lower() or "mbengue" in f.lower():
                chemin_alphonse = f"dev_data/{f}"
                print(f"✅ Fichier trouvé: {f}")
                break
        else:
            # Prendre le premier fichier si Alphonse non trouvé
            fichiers = [f for f in os.listdir("dev_data") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if fichiers:
                chemin_alphonse = f"dev_data/{fichiers[0]}"
                print(f"⚠️  Utilisation du fichier: {fichiers[0]}")
            else:
                print("❌ Aucun fichier trouvé dans dev_data/")
                return
        
        image = cv2.imread(chemin_alphonse)
        if image is not None:
            # Détecter le visage d'Alphonse
            gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detecteur = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            visages = detecteur.detectMultiScale(gris, 1.1, 4)
            
            if len(visages) > 0:
                x, y, w, h = visages[0]
                self.visage_alphonse = gris[y:y+h, x:x+w]
                self.visage_alphonse = cv2.resize(self.visage_alphonse, (100, 100))
                print("✅ Visage de référence chargé")
            else:
                print("❌ Aucun visage détecté dans la photo")
        else:
            print("❌ Impossible de charger l'image")
    
    def comparer_avec_alphonse(self, visage_capture):
        """Compare un visage avec la référence"""
        if self.visage_alphonse is None:
            return 0.0
        
        try:
            # Redimensionner à la même taille
            visage_capture = cv2.resize(visage_capture, (100, 100))
            
            # Méthode simple de comparaison
            difference = cv2.absdiff(visage_capture, self.visage_alphonse)
            score = 1 - (np.mean(difference) / 255.0)
            
            return score
        except:
            return 0.0
    
    def est_alphonse(self, image):
        """Détermine si l'image contient Alphonse"""
        # Détection du visage
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detecteur = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        visages = detecteur.detectMultiScale(gris, 1.1, 4, minSize=(100, 100))
        
        if len(visages) == 0:
            return False, 0.0
        
        # Extraire le visage
        x, y, w, h = visages[0]
        visage_capture = gris[y:y+h, x:x+w]
        
        # Comparer avec Alphonse
        score = self.comparer_avec_alphonse(visage_capture)
        
        return score > 0.6, score  # Seuil de 60%
    
    def sauvegarder_pointage(self, score):
        """Sauvegarde un pointage"""
        heure = datetime.now().strftime("%H:%M:%S")
        pointage = f"Alphonse Marie Mbengue - {score:.1%} - {heure}"
        self.pointages.append(pointage)
        print(f"✅ {pointage}")
        return pointage
    
    def executer(self):
        """Boucle principale de test"""
        if self.visage_alphonse is None:
            print("❌ Impossible de continuer sans référence")
            return
        
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera = cv2.VideoCapture(1)
        
        if not camera.isOpened():
            print("❌ Aucune caméra")
            return
        
        print("🚀 Test de reconnaissance démarré!")
        print("📍 Placez-vous devant la caméra")
        print("🎮 Q pour quitter")
        
        dernier_pointage = None
        
        try:
            while True:
                succes, image = camera.read()
                if not succes:
                    break
                
                image = cv2.flip(image, 1)
                
                # Détection visage pour l'interface
                gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                detecteur = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
                visages = detecteur.detectMultiScale(gris, 1.1, 4)
                
                est_alphonse = False
                score = 0.0
                
                # Reconnaissance
                if len(visages) == 1:
                    est_alphonse, score = self.est_alphonse(image)
                    
                    # Pointage automatique
                    if est_alphonse and score > 0.6:
                        if dernier_pointage is None or time.time() - dernier_pointage > 10:
                            self.sauvegarder_pointage(score)
                            dernier_pointage = time.time()
                
                # Interface - CORRECTION ICI
                for (x, y, w, h) in visages:
                    if est_alphonse:
                        couleur = (0, 255, 0)  # Vert
                        texte = f"ALPHONSE ({score:.0%})"
                    else:
                        couleur = (0, 255, 255)  # Jaune
                        texte = "AUTRE PERSONNE"
                    
                    cv2.rectangle(image, (x, y), (x+w, y+h), couleur, 3)
                    
                    # Bandeau d'identification
                    cv2.rectangle(image, (x, y-40), (x+w, y), couleur, -1)
                    cv2.putText(image, texte, (x+5, y-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # En-tête
                cv2.putText(image, "MANGUI FI - RECONNAISSANCE", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Statut - CORRECTION ICI
                if est_alphonse:
                    statut = f"ALPHONSE DETECTE - Confiance: {score:.1%}"
                    couleur_statut = (0, 255, 0)
                elif len(visages) > 0:  # CORRIGÉ: vérifier la longueur du tableau
                    statut = "AUTRE PERSONNE - Alphonse non reconnu"
                    couleur_statut = (0, 165, 255)
                else:
                    statut = "EN ATTENTE..."
                    couleur_statut = (255, 255, 255)
                
                cv2.putText(image, statut, (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur_statut, 1)
                
                cv2.imshow('Test Reconnaissance - MANGUI FI', image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n🛑 Arrêt demandé")
        except Exception as e:
            print(f"💥 Erreur: {e}")
        finally:
            camera.release()
            cv2.destroyAllWindows()
            
            print("=" * 40)
            print("📊 TEST TERMINE")
            print(f"Pointages: {len(self.pointages)}")
            for p in self.pointages:
                print(f"  • {p}")

# Lancement
if __name__ == "__main__":
    test = TestAlphonse()
    test.executer()
