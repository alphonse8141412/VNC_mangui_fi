#!/usr/bin/env python3
"""
TERMINAL MANGUI FI - VERSION SIMPLE
"""

import cv2
import time
from datetime import datetime

print("🎯 MANGUI FI - SYSTEME DE POINTAGE")
print("=" * 40)

# 1. Initialisation caméra
print("1. 📹 Initialisation caméra...")
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("❌ ERREUR: Aucune caméra branchée")
    exit()

print("✅ Caméra OK")

# 2. Chargement détecteur
print("2. 🔍 Chargement détecteur...")
detecteur = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if detecteur.empty():
    print("❌ ERREUR: Détecteur non chargé")
    camera.release()
    exit()

print("✅ Détecteur OK")

# 3. Configuration
agents = ["ALPHA", "BETA", "GAMMA", "DELTA"]
pointages = []

print("3. ✅ Système prêt")
print(f"   Agents: {len(agents)}")
print("   📋 Contrôles: Q=Quitter, P=Pointage")
print("=" * 40)

# 4. Boucle principale
dernier_pointage = None

while True:
    # Capture image
    succes, image = camera.read()
    if not succes:
        break
    
    # Miroir
    image = cv2.flip(image, 1)
    
    # Détection visages
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    visages = detecteur.detectMultiScale(gris, 1.1, 4)
    
    # Affichage résultats
    for (x, y, w, h) in visages:
        # Rectangle vert
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Texte
        cv2.putText(image, "VISAGE DETECTE", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Pointage automatique
    if len(visages) == 1:
        if dernier_pointage is None or time.time() - dernier_pointage > 5:
            heure = datetime.now().strftime("%H:%M:%S")
            agent = agents[0]
            pointages.append(f"{agent} - {heure}")
            print(f"✅ POINTAGE AUTO: {agent} à {heure}")
            dernier_pointage = time.time()
    
    # Interface
    cv2.putText(image, "MANGUI FI - POINTAGE", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(image, f"Visages: {len(visages)}", (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(image, "Q=Quitter  P=Pointage Manuel", (20, 450), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Affichage
    cv2.imshow('MANGUI FI - Terminal', image)
    
    # Contrôles
    touche = cv2.waitKey(1) & 0xFF
    if touche == ord('q'):
        break
    elif touche == ord('p') and len(visages) > 0:
        # Pointage manuel
        heure = datetime.now().strftime("%H:%M:%S")
        agent = agents[0]
        pointages.append(f"{agent} - {heure}")
        print(f"✅ POINTAGE MANUEL: {agent} à {heure}")
        dernier_pointage = time.time()

# Nettoyage
camera.release()
cv2.destroyAllWindows()

# Résumé
print("=" * 40)
print(f"📊 Session terminée")
print(f"📋 Total pointages: {len(pointages)}")
for p in pointages:
    print(f"   • {p}")
print("👋 Au revoir!")
