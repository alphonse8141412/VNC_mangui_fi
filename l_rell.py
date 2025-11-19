#!/usr/bin/env python3
"""
MANGUI FI - SYSTÈME POUR 5 PERSONNES
Version avec les noms réels
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
        self.references_encodings = []
        self.noms_references = []
        self.derniers_pointages = {}
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
        self.nom_fenetre = 'MANGUI FI - 5 PERSONNES'
        
        self.charger_references_multiple()

    def charger_references_multiple(self):
        """Charge les références pour les 5 personnes spécifiques"""
        try:
            # Dossier contenant les photos de référence
            dossier_references = "/home/alphonse/facialVCN/VNC_mangui_fi/marie/"
            
            # Liste des personnes avec leurs fichiers exacts
            personnes = [
                {"nom": "ALLA NIANG", "fichier": "Alla NIANG.jpg"},
                {"nom": "ALPHONSE MARIE MBENGUE", "fichier": "Alphonse Marie Mbengue.jpg"},
                {"nom": "AMINATA NIANG", "fichier": "Aminata Niang.jpg"},
                {"nom": "ASSANE DIONE", "fichier": "Assane Dione.jpg"},
                {"nom": "YOUSSOUPHA SY", "fichier": "YOUSSOUPHA-SY.jpg,"}
            ]
            
            print("📸 Chargement des références pour 5 personnes...")
            
            for personne in personnes:
                chemin_ref = os.path.join(dossier_references, personne["fichier"])
                
                if not os.path.exists(chemin_ref):
                    print(f"⚠️  Photo non trouvée: {personne['fichier']}")
                    continue
                
                print(f"   Chargement: {personne['nom']}...")
                
                image_bgr = cv2.imread(chemin_ref)
                if image_bgr is None:
                    print(f"❌ Impossible de charger: {personne['fichier']}")
                    continue
                
                # Redimensionner si nécessaire
                max_size = 1000
                height, width = image_bgr.shape[:2]
                if height > max_size or width > max_size:
                    scale = max_size / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image_bgr = cv2.resize(image_bgr, (new_width, new_height))
                    print(f"     Redimensionné à: {new_width}x{new_height}")
                
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                
                # Détection du visage
                face_locations = face_recognition.face_locations(image_rgb, model="hog")
                
                if face_locations:
                    encodings = face_recognition.face_encodings(image_rgb, face_locations)
                    if encodings:
                        self.references_encodings.append(encodings[0])
                        self.noms_references.append(personne["nom"])
                        self.derniers_pointages[personne["nom"]] = 0
                        print(f"     ✅ {personne['nom']} - Référence chargée")
                    else:
                        print(f"     ❌ Impossible d'encoder: {personne['nom']}")
                else:
                    print(f"     ❌ Aucun visage détecté pour: {personne['nom']}")
            
            print(f"\n✅ CHARGEMENT TERMINÉ: {len(self.references_encodings)} références chargées sur 5")
            
            # Afficher le résumé
            if self.noms_references:
                print("👥 PERSONNES CHARGÉES:")
                for i, nom in enumerate(self.noms_references, 1):
                    print(f"   {i}. {nom}")
                    
        except Exception as e:
            print(f"❌ Erreur chargement références: {e}")

    def initialiser_camera(self):
        """Initialise la caméra et la fenêtre d'affichage"""
        print("📷 Initialisation caméra et affichage...")
        
        # Créer la fenêtre AVANT d'initialiser la caméra
        cv2.namedWindow(self.nom_fenetre, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.nom_fenetre, self.taille_affichage[0], self.taille_affichage[1])
        cv2.moveWindow(self.nom_fenetre, 100, 100)
        
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
                    cv2.imshow(self.nom_fenetre, test_frame)
                    cv2.waitKey(100)
                else:
                    print("❌ Caméra ne renvoie pas d'image")
                    cap.release()
                    continue
                    
                return cap
            cap.release()
        
        print("❌ Aucune caméra fonctionnelle trouvée")
        return None

    def detecter_et_reconnaitre(self, frame):
        """Détection et reconnaissance pour plusieurs personnes"""
        face_locations = []
        noms = []
        
        try:
            # Détection sur résolution réduite
            small_frame = cv2.resize(frame, self.taille_traitement)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            
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
            
            # Encodage des visages détectés
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Reconnaissance pour chaque visage
            noms = []
            for face_encoding in face_encodings:
                nom, couleur = self.comparer_visage_multiple(face_encoding)
                noms.append((nom, couleur))
            
            return face_locations_fullres, noms
            
        except Exception as e:
            print(f"⚠️  Erreur détection: {e}")
            return [], []

    def comparer_visage_multiple(self, face_encoding):
        """Compare un visage avec toutes les références"""
        if not self.references_encodings:
            return "INCONNU", (0, 0, 255)
        
        try:
            # Calcul des distances avec toutes les références
            distances = face_recognition.face_distance(self.references_encodings, face_encoding)
            
            # Trouver la meilleure correspondance
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            confidence = 1.0 - best_distance
            
            nom_trouve = self.noms_references[best_match_index]
            print(f"   🔍 {nom_trouve} (confiance: {confidence:.3f})")
            
            if confidence > 0.6:
                couleur = self.get_couleur_personne(nom_trouve)
                
                # Pointage automatique
                temps_actuel = time.time()
                if temps_actuel - self.derniers_pointages[nom_trouve] > 30:
                    self.sauvegarder_pointage(nom_trouve, confidence)
                    self.derniers_pointages[nom_trouve] = temps_actuel
                    
                return nom_trouve, couleur
            else:
                return f"INCONNU ({confidence:.2f})", (0, 0, 255)
                
        except Exception as e:
            print(f"❌ Erreur comparaison: {e}")
            return "ERREUR", (255, 0, 0)

    def get_couleur_personne(self, nom):
        """Retourne une couleur spécifique pour chaque personne"""
        couleurs = {
            "ALLA NIANG": (0, 255, 0),           # Vert
            "ALPHONSE MARIE MBENGUE": (255, 0, 0), # Bleu
            "AMINATA NIANG": (0, 255, 255),      # Jaune
            "ASSANE DIONE": (255, 0, 255),       # Magenta
            "YOUSSOUPHA SY": (255, 255, 0)       # Cyan
        }
        return couleurs.get(nom, (0, 0, 255))  # Rouge par défaut

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
            
            # Anti-doublon pour la même personne
            derniers_pointages_personne = [p for p in pointages[-10:] if p['agent'] == nom]
            if derniers_pointages_personne:
                dernier = derniers_pointages_personne[-1]
                if time.time() - dernier['timestamp'] < 25:
                    return
            
            pointages.append(pointage)
            
            with open(self.pointages_file, 'w') as f:
                json.dump(pointages, f, indent=2)
            
            print(f"✅ POINTAGE: {nom} à {pointage['heure']} (confiance: {confidence:.2f})")
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")

    def executer(self):
        """Lance le système pour 5 personnes"""
        print("🎯 MANGUI FI - SYSTÈME 5 PERSONNES")
        print("=" * 50)
        
        if self.references_encodings:
            print(f"✅ {len(self.references_encodings)} personnes chargées")
        else:
            print("⚠️  Aucune référence chargée - Mode détection seulement")
        
        cap = self.initialiser_camera()
        if cap is None:
            print("❌ Impossible de démarrer sans caméra")
            return
        
        print("✅ Système initialisé")
        print("📍 Contrôles: Q=Quitter, P=Pointage, S=Stats, L=Liste personnes")
        print("👀 Vérifiez l'affichage de la caméra...")
        
        time.sleep(1)
        
        try:
            while True:
                debut = time.time()
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Erreur capture - Caméra déconnectée?")
                    break
                
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
                
                # Affichage
                self.afficher_resultats(frame)
                
                # Contrôles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.pointage_manuel()
                elif key == ord('s'):
                    self.afficher_statistiques()
                elif key == ord('l'):
                    self.afficher_liste_personnes()
                
                self.compteur_frames += 1
                
                if self.compteur_frames % 100 == 0:
                    print(f"📊 Frame {self.compteur_frames} - Système actif")
                
                temps_frame = time.time() - debut
                if temps_frame < 0.1:
                    time.sleep(0.1 - temps_frame)
                    
        except KeyboardInterrupt:
            print("\n🛑 Arrêt demandé")
        except Exception as e:
            print(f"❌ Erreur système: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
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
        """Interface utilisateur pour 5 personnes"""
        h, w = frame.shape[:2]
        nb_visages = len(self.derniers_visages)
        noms = [nom for nom, _ in self.derniers_noms]
        
        # En-tête semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Statut références
        statut_ref = f"PERSONNES: {len(self.references_encodings)}/5"
        cv2.putText(frame, statut_ref, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Statut principal
        if nb_visages > 0:
            personnes_reconnues = [nom for nom in noms if nom in self.noms_references]
            if personnes_reconnues:
                if len(personnes_reconnues) == 1:
                    statut = f"{personnes_reconnues[0]} - RECONNU"
                else:
                    statut = f"{len(personnes_reconnues)} PERSONNES RECONNUES"
                couleur_statut = (0, 255, 0)
            else:
                statut = f"{nb_visages} INCONNU(S)"
                couleur_statut = (0, 165, 255)
        else:
            statut = "EN ATTENTE DE DETECTION..."
            couleur_statut = (255, 255, 255)
        
        cv2.putText(frame, "MANGUI FI - 5 PERSONNES", (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, statut, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur_statut, 1)
        
        # Informations
        info_text = f"Frame: {self.compteur_frames} | Visages: {nb_visages}"
        cv2.putText(frame, info_text, (w - 250, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Pied de page
        cv2.rectangle(frame, (0, h-30), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "Q=Quitter  P=Pointage  S=Stats  L=Liste", (10, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def pointage_manuel(self):
        """Pointage manuel"""
        if self.derniers_visages:
            try:
                noms_detectes = [nom for nom, _ in self.derniers_noms]
                if noms_detectes and noms_detectes[0] in self.noms_references:
                    nom = noms_detectes[0]
                    self.sauvegarder_pointage(f"{nom} (manuel)", 0.99)
                    print(f"✅ Pointage manuel pour {nom}")
                else:
                    print("❌ Aucune personne reconnue pour pointage manuel")
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
                
                print(f"\n📊 STATISTIQUES MANGUI FI:")
                print(f"   Pointages aujourd'hui: {len(pointages_auj)}")
                print(f"   Total historique: {len(pointages)}")
                
                # Statistiques par personne
                if pointages_auj:
                    print(f"   Détail aujourd'hui:")
                    for personne in self.noms_references:
                        count = len([p for p in pointages_auj if p['agent'] == personne])
                        if count > 0:
                            print(f"     - {personne}: {count} pointages")
                
                if pointages_auj:
                    print(f"   Derniers pointages:")
                    for p in pointages_auj[-5:]:
                        print(f"     - {p['heure']} ({p['agent']})")
            else:
                print("📊 Aucun pointage enregistré")
        except Exception as e:
            print(f"❌ Erreur stats: {e}")

    def afficher_liste_personnes(self):
        """Affiche la liste des personnes enregistrées"""
        print(f"\n👥 LISTE DES PERSONNES ENREGISTRÉES ({len(self.noms_references)}/5):")
        for i, nom in enumerate(self.noms_references, 1):
            couleur = self.get_couleur_personne(nom)
            couleur_nom = "Vert" if couleur == (0, 255, 0) else \
                         "Bleu" if couleur == (255, 0, 0) else \
                         "Jaune" if couleur == (0, 255, 255) else \
                         "Magenta" if couleur == (255, 0, 255) else \
                         "Cyan" if couleur == (255, 255, 0) else "Rouge"
            print(f"   {i}. {nom} - Couleur: {couleur_nom}")

# Lancement du système
if __name__ == "__main__":
    print("🚀 Démarrage MANGUI FI - Système 5 Personnes...")
    systeme = SystemeReconnaissanceFaciale()
    systeme.executer()

