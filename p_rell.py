#!/usr/bin/env python3
"""
MANGUI FI - SYSTÈME AVEC AFFICHAGE CAMÉRA CORRIGÉ
Version avec gestion d'affichage garantie
"""

import cv2
import face_recognition
import numpy as np
import time
import json
import os
from datetime import datetime

class SystemeReconnaissanceFaciale:
    def __init__(self):
        self.camera_index = 0
        self.pointages_file = "pointages_manguifi.json"
        self.reference_encoding = None
        self.dernier_pointage = 0
        self.compteur_frames = 0
        self.frame_skip = 3
        
        # Résolutions
        self.taille_traitement = (320, 240)
        self.taille_affichage = (640, 480)
        
        # Stockage des dernières détections
        self.derniers_visages = []
        self.derniers_noms = []
        self.derniere_detection = 0
        
        # Configuration fenêtre
        self.nom_fenetre = 'MANGUI FI - RECONNAISSANCE FACIALE'
        
        self.charger_reference_rapide()

    def charger_reference_rapide(self):
        """Charge la référence rapidement avec redimensionnement"""
        try:
            chemin_ref = "/home/alphonse/facialVCN/VNC_mangui_fi/marie/Alphonse Marie Mbengue.jpg"
            if not os.path.exists(chemin_ref):
                print("❌ Photo référence non trouvée")
                return

            print("📸 Chargement RAPIDE de la référence...")
            
            image_bgr = cv2.imread(chemin_ref)
            if image_bgr is None:
                print("❌ Impossible de charger l'image")
                return
                
            print(f"   Taille originale: {image_bgr.shape}")
            
            # Redimensionner
            max_size = 1000
            height, width = image_bgr.shape[:2]
            
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_bgr = cv2.resize(image_bgr, (new_width, new_height))
                print(f"   Redimensionné à: {image_bgr.shape}")
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(
                image_rgb, 
                number_of_times_to_upsample=0,
                model="hog"
            )
            
            if face_locations:
                print(f"   ✅ {len(face_locations)} visage(s) détecté(s)")
                encodings = face_recognition.face_encodings(image_rgb, face_locations)
                if encodings:
                    self.reference_encoding = encodings[0]
                    print("✅ Référence encodée avec succès")
            else:
                print("❌ Aucun visage détecté")
                    
        except Exception as e:
            print(f"❌ Erreur chargement: {e}")

    def initialiser_camera(self):
        """Initialise la caméra et la fenêtre d'affichage"""
        print("📷 Initialisation caméra et affichage...")
        
        # Créer la fenêtre AVANT d'initialiser la caméra
        cv2.namedWindow(self.nom_fenetre, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.nom_fenetre, self.taille_affichage[0], self.taille_affichage[1])
        cv2.moveWindow(self.nom_fenetre, 100, 100)  # Position sur l'écran
        
        for i in [0, 1, 2]:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✅ Caméra trouvée sur l'index {i}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.taille_affichage[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.taille_affichage[1])
                cap.set(cv2.CAP_PROP_FPS, 15)
                
                # Tester l'affichage immédiatement
                ret, test_frame = cap.read()
                if ret:
                    print("✅ Caméra fonctionnelle - Test d'affichage...")
                    # Afficher un frame de test
                    cv2.imshow(self.nom_fenetre, test_frame)
                    cv2.waitKey(100)  # Court délai pour l'affichage
                else:
                    print("❌ Caméra ne renvoie pas d'image")
                    cap.release()
                    continue
                    
                return cap
            cap.release()
        
        print("❌ Aucune caméra fonctionnelle trouvée")
        return None

    def detecter_et_reconnaitre(self, frame):
        """Détection et reconnaissance"""
        face_locations = []
        noms = []
        
        try:
            # Détection sur résolution réduite
            small_frame = cv2.resize(frame, self.taille_traitement)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(
                rgb_small_frame, 
                number_of_times_to_upsample=0,
                model="hog"
            )
            
            if not face_locations:
                return [], []
            
            # Conversion coordonnées
            face_locations_fullres = []
            for (top, right, bottom, left) in face_locations:
                scale_y = self.taille_affichage[1] / self.taille_traitement[1]
                scale_x = self.taille_affichage[0] / self.taille_traitement[0]
                
                top = int(top * scale_y)
                right = int(right * scale_x)
                bottom = int(bottom * scale_y)
                left = int(left * scale_x)
                
                face_locations_fullres.append((top, right, bottom, left))
            
            # Encodage
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Reconnaissance
            noms = []
            for face_encoding in face_encodings:
                nom, couleur = self.comparer_visage(face_encoding)
                noms.append((nom, couleur))
            
            return face_locations_fullres, noms
            
        except Exception as e:
            print(f"⚠️  Erreur détection: {e}")
            return [], []

    def comparer_visage(self, face_encoding):
        """Compare un visage avec la référence"""
        if self.reference_encoding is None:
            return "INCONNU", (0, 0, 255)
        
        try:
            distances = face_recognition.face_distance([self.reference_encoding], face_encoding)
            distance_val = float(distances[0])
            confidence = 1.0 - distance_val
            
            print(f"   🔍 Confiance: {confidence:.3f}")
            
            if confidence > 0.6:
                nom = "ALPHONSE MARIE MBENGUE"
                couleur = (0, 255, 0)
                
                # Pointage automatique
                temps_actuel = time.time()
                if temps_actuel - self.dernier_pointage > 30:
                    self.sauvegarder_pointage(nom, confidence)
                    self.dernier_pointage = temps_actuel
            else:
                nom = f"INCONNU ({confidence:.2f})"
                couleur = (0, 0, 255)
                
            return nom, couleur
            
        except Exception as e:
            print(f"❌ Erreur comparaison: {e}")
            return "ERREUR", (255, 0, 0)

    def sauvegarder_pointage(self, nom, confidence=1.0):
        """Sauvegarde des pointages"""
        pointage = {
            'agent': nom,
            'heure': datetime.now().strftime("%H:%M:%S"),
            'date': datetime.now().strftime("%Y-%m-%d"),
            'confidence': f"{confidence:.2f}",
            'timestamp': time.time()
        }
        
        try:
            pointages = []
            if os.path.exists(self.pointages_file):
                with open(self.pointages_file, 'r') as f:
                    pointages = json.load(f)
            
            if pointages:
                dernier = pointages[-1]
                if time.time() - dernier['timestamp'] < 25:
                    return
            
            pointages.append(pointage)
            
            with open(self.pointages_file, 'w') as f:
                json.dump(pointages, f, indent=2)
            
            print(f"✅ POINTAGE: {nom} à {pointage['heure']} (confiance: {confidence:.2f})")
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")

    def executer(self):
        """Lance le système avec affichage garanti"""
        print("🎯 MANGUI FI - AFFICHAGE CAMÉRA GARANTI")
        print("=" * 50)
        
        if self.reference_encoding is not None:
            print("✅ Référence chargée - Reconnaissance activée")
        else:
            print("⚠️  Mode détection seulement")
        
        cap = self.initialiser_camera()
        if cap is None:
            print("❌ Impossible de démarrer sans caméra")
            return
        
        print("✅ Système initialisé")
        print("📍 Contrôles: Q=Quitter, P=Pointage, S=Stats")
        print("👀 Vérifiez l'affichage de la caméra...")
        
        # Attendre un peu pour que la fenêtre s'affiche
        time.sleep(1)
        
        try:
            while True:
                debut = time.time()
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Erreur capture - Caméra déconnectée?")
                    break
                
                # Vérifier que l'image n'est pas vide
                if frame is None or frame.size == 0:
                    print("❌ Image vide de la caméra")
                    continue
                
                # Traitement tous les N frames
                if self.compteur_frames % self.frame_skip == 0:
                    try:
                        face_locations, noms = self.detecter_et_reconnaitre(frame)
                        
                        if face_locations:
                            self.derniers_visages = face_locations
                            self.derniers_noms = noms
                            self.derniere_detection = time.time()
                        else:
                            if time.time() - self.derniere_detection > 2.0:
                                self.derniers_visages = []
                                self.derniers_noms = []
                                
                    except Exception as e:
                        print(f"⚠️  Erreur traitement: {e}")
                
                # TOUJOURS afficher le frame même sans détection
                self.afficher_resultats(frame)
                
                # AFFICHAGE CRITIQUE - Utiliser waitKey correctement
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.pointage_manuel()
                elif key == ord('s'):
                    self.afficher_statistiques()
                
                self.compteur_frames += 1
                
                # Feedback visuel toutes les 100 frames
                if self.compteur_frames % 100 == 0:
                    print(f"📊 Frame {self.compteur_frames} - Système actif")
                
                # Limiter FPS
                temps_frame = time.time() - debut
                if temps_frame < 0.1:
                    time.sleep(0.1 - temps_frame)
                    
        except KeyboardInterrupt:
            print("\n�� Arrêt demandé")
        except Exception as e:
            print(f"❌ Erreur système: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # S'assurer que les fenêtres sont fermées
            print("👋 Système arrêté")

    def afficher_resultats(self, frame):
        """Affiche les résultats avec gestion d'erreur d'affichage"""
        try:
            # Dessiner les rectangles de détection
            for (top, right, bottom, left), (nom, couleur) in zip(self.derniers_visages, self.derniers_noms):
                cv2.rectangle(frame, (left, top), (right, bottom), couleur, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), couleur, cv2.FILLED)
                cv2.putText(frame, nom, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Interface utilisateur
            self.afficher_interface(frame)
            
            # AFFICHAGE PRINCIPAL
            cv2.imshow(self.nom_fenetre, frame)
            
        except Exception as e:
            print(f"❌ Erreur affichage: {e}")

    def afficher_interface(self, frame):
        """Interface utilisateur"""
        h, w = frame.shape[:2]
        nb_visages = len(self.derniers_visages)
        noms = [nom for nom, _ in self.derniers_noms]
        
        # En-tête semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Statut référence
        statut_ref = "REF: ✅" if self.reference_encoding is not None else "REF: ❌"
        cv2.putText(frame, statut_ref, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Statut principal
        if nb_visages > 0:
            if "ALPHONSE MARIE MBENGUE" in noms:
                statut = "ALPHONSE - POINTAGE AUTO"
                couleur_statut = (0, 255, 0)
            else:
                statut = f"{nb_visages} INCONNU(S)"
                couleur_statut = (0, 165, 255)
        else:
            statut = "EN ATTENTE..."
            couleur_statut = (255, 255, 255)
        
        cv2.putText(frame, "MANGUI FI - RECONNAISSANCE", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, statut, (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur_statut, 1)
        
        # Informations
        info_text = f"Frame: {self.compteur_frames} | Visages: {nb_visages}"
        cv2.putText(frame, info_text, (w - 250, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Pied de page
        cv2.rectangle(frame, (0, h-25), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "Q=Quitter  P=Pointage  S=Stats", (10, h-8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def pointage_manuel(self):
        """Pointage manuel"""
        if self.derniers_visages:
            try:
                nom = "ALPHONSE MARIE MBENGUE" if self.reference_encoding is not None else "VISAGE DETECTE"
                self.sauvegarder_pointage(f"{nom} (manuel)", 0.99)
                print("✅ Pointage manuel enregistré")
            except Exception as e:
                print(f"❌ Erreur pointage: {e}")
        else:
            print("❌ Aucun visage détecté pour pointage manuel")

    def afficher_statistiques(self):
        """Affiche les statistiques"""
        try:
            if os.path.exists(self.pointages_file):
                with open(self.pointages_file, 'r') as f:
                    pointages = json.load(f)
                
                aujourd_hui = datetime.now().strftime("%Y-%m-%d")
                pointages_auj = [p for p in pointages if p['date'] == aujourd_hui]
                
                print(f"\n📊 STATISTIQUES:")
                print(f"   Aujourd'hui: {len(pointages_auj)}")
                print(f"   Total: {len(pointages)}")
                
                if pointages_auj:
                    print(f"   Derniers:")
                    for p in pointages_auj[-3:]:
                        print(f"     - {p['heure']} ({p['agent']})")
            else:
                print("📊 Aucun pointage")
        except Exception as e:
            print(f"❌ Erreur stats: {e}")

# Lancement du système
if __name__ == "__main__":
    print("🚀 Démarrage MANGUI FI - Affichage Garanti...")
    systeme = SystemeReconnaissanceFaciale()
    systeme.executer()
