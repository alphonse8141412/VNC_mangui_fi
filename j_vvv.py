#!/usr/bin/env python3
"""
MANGUI FI - SYSTÈME AMÉLIORÉ AVEC VERROUILLAGE
Version avec couleurs simplifiées, masquage des analyses et verrouillage temporel
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
        
        # --- NOUVEAU : Variables pour le verrouillage ---
        self.personne_verrouillee = None          # Nom de la personne actuellement verrouillée
        self.position_verrouillee = None          # Position du rectangle verrouillé
        self.temps_fin_verrouillage = 0           # Timestamp de fin des 2 minutes
        self.validation_compteur = 0              # Compteur pour la validation de 5 secondes
        self.validation_nom = None                # Nom en cours de validation
        self.validation_requise = 15              # ~5 secondes (15 frames à 3 FPS de traitement)
        self.derniere_validation_time = 0         # Dernier temps de validation
        
        # Configuration fenêtre
        self.nom_fenetre = 'MANGUI FI - SYSTÈME VERROUILLÉ'
        
        self.charger_references_multiple()

    def charger_references_multiple(self):
        """Charge les références pour les 7 personnes spécifiques"""
        try:
            # Dossier contenant les photos de référence
            dossier_references = "/home/alphonse/facialVCN/VNC_mangui_fi/marie/"
            
            # Liste des personnes avec leurs fichiers exacts
            personnes = [
                {"nom": "ALLA NIANG", "fichier": "Alla NIANG.jpg"},
                {"nom": "ALPHONSE MARIE MBENGUE", "fichier": "Alphonse Marie Mbengue.jpg"},
                {"nom": "AMINATA NIANG", "fichier": "Aminata Niang.jpg"},
                {"nom": "ASSANE DIONE", "fichier": "Assane Dione.jpg"},
                {"nom": "YOUSSOUPHA SY", "fichier": "YOUSSOUPHA-SY.jpg"},
                {"nom": "FALLOU DIOP", "fichier": "Fallou Diop.jpg"},
                {"nom": "EL HADJI MALICK", "fichier": "El Hadji Malick Ndiaye_.jpg"}
            ]
            
            print("📸 Chargement des références pour 7 personnes...")
            
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
            
            print(f"\n✅ CHARGEMENT TERMINÉ: {len(self.references_encodings)} références chargées sur 7")
            
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
        """Compare un visage avec toutes les références - VERSION SIMPLIFIÉE"""
        if not self.references_encodings:
            return "INCONNU", (0, 0, 255)  # Rouge pour inconnu
        
        try:
            # Calcul des distances avec toutes les références
            distances = face_recognition.face_distance(self.references_encodings, face_encoding)
            
            # Trouver la meilleure correspondance
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            confidence = 1.0 - best_distance
            
            nom_trouve = self.noms_references[best_match_index]
            
            # SUPPRIMÉ: Ligne de débogage pour masquer l'analyse
            # print(f"   🔍 {nom_trouve} (confiance: {confidence:.3f})")
            
            # LOGIQUE SIMPLIFIÉE : Vert si reconnu, Rouge sinon
            if confidence > 0.6:
                return nom_trouve, (0, 255, 0)  # VERT pour reconnu
            else:
                return f"INCONNU", (0, 0, 255)  # ROUGE pour inconnu
                
        except Exception as e:
            print(f"❌ Erreur comparaison: {e}")
            return "ERREUR", (255, 0, 0)

    def gerer_verrouillage_et_validation(self, face_locations, noms):
        """Gère la logique de validation et verrouillage - NOUVELLE MÉTHODE"""
        temps_actuel = time.time()
        
        # PHASE 1: Vérifier si on est en mode verrouillé
        if self.personne_verrouillee and temps_actuel < self.temps_fin_verrouillage:
            # Mode verrouillé actif - Maintenir l'affichage du rectangle vert
            temps_restant = int(self.temps_fin_verrouillage - temps_actuel)
            
            # Chercher si la personne verrouillée est toujours détectée
            personne_trouvee = False
            for (top, right, bottom, left), (nom, couleur) in zip(face_locations, noms):
                if nom == self.personne_verrouillee:
                    personne_trouvee = True
                    self.position_verrouillee = (top, right, bottom, left)
                    break
            
            # Si la personne n'est pas trouvée, garder la dernière position connue
            if not personne_trouvee and self.position_verrouillee:
                face_locations = [self.position_verrouillee]
                noms = [(self.personne_verrouillee + f" ({temps_restant}s)", (0, 255, 0))]
            elif personne_trouvee:
                # Mettre à jour l'affichage avec le temps restant
                noms = [(self.personne_verrouillee + f" ({temps_restant}s)", (0, 255, 0))]
            
            return face_locations, noms
        
        # PHASE 2: Si le verrouillage est terminé, réinitialiser
        elif self.personne_verrouillee and temps_actuel >= self.temps_fin_verrouillage:
            print(f"🔓 Fin du verrouillage pour {self.personne_verrouillee}")
            self.personne_verrouillee = None
            self.position_verrouillee = None
            self.validation_compteur = 0
            self.validation_nom = None
        
        # PHASE 3: Validation des nouvelles détections
        if face_locations and not self.personne_verrouillee:
            # Prendre le premier visage détecté pour la validation
            premier_nom, premiere_couleur = noms[0]
            premiere_position = face_locations[0]
            
            # Vérifier si c'est une personne reconnue (VERT)
            if premiere_couleur == (0, 255, 0):  # Vert = reconnu
                if self.validation_nom == premier_nom:
                    # Même personne - Incrémenter le compteur de validation
                    self.validation_compteur += 1
                    
                    # Vérifier si la validation est complète (5 secondes)
                    if self.validation_compteur >= self.validation_requise:
                        print(f"✅ VALIDATION TERMINÉE: {premier_nom} - Pointage automatique")
                        self.sauvegarder_pointage(premier_nom, 0.85)  # Confiance élevée pour validation
                        
                        # Activer le verrouillage pour 2 minutes
                        self.personne_verrouillee = premier_nom
                        self.position_verrouillee = premiere_position
                        self.temps_fin_verrouillage = temps_actuel + 120  # 2 minutes
                        self.validation_compteur = 0
                        self.validation_nom = None
                        
                        # Mettre à jour l'affichage pour le mode verrouillé
                        face_locations = [premiere_position]
                        noms = [(premier_nom + " (VERROUILLÉ)", (0, 255, 0))]
                        
                        return face_locations, noms
                else:
                    # Nouvelle personne - Démarrer un nouveau cycle de validation
                    self.validation_nom = premier_nom
                    self.validation_compteur = 1
                    self.derniere_validation_time = temps_actuel
                    print(f"🔄 Début validation: {premier_nom} (1/{self.validation_requise})")
            else:
                # Personne non reconnue ou INCONNU - Réinitialiser la validation
                if self.validation_nom:
                    print(f"❌ Validation interrompue: visage non reconnu")
                    self.validation_compteur = 0
                    self.validation_nom = None
        
        # PHASE 4: Afficher le statut de validation en cours
        if self.validation_nom and not self.personne_verrouillee:
            progression = f"({self.validation_compteur}/{self.validation_requise})"
            for i, (nom, couleur) in enumerate(noms):
                if nom == self.validation_nom:
                    noms[i] = (f"{nom} {progression}", (0, 255, 0))  # Vert pour validation
        
        return face_locations, noms

    def sauvegarder_pointage(self, nom, confidence=1.0):
        """Sauvegarde des pointages avec anti-doublon amélioré"""
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
            
            # Anti-doublon amélioré : 30 secondes minimum entre les pointages
            derniers_pointages_personne = [p for p in pointages[-10:] if p['agent'] == nom]
            if derniers_pointages_personne:
                dernier = derniers_pointages_personne[-1]
                if time.time() - dernier['timestamp'] < 30:
                    return
            
            pointages.append(pointage)
            
            with open(self.pointages_file, 'w') as f:
                json.dump(pointages, f, indent=2)
            
            print(f"✅ POINTAGE: {nom} à {pointage['heure']}")
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")

    def executer(self):
        """Lance le système avec la nouvelle logique de verrouillage"""
        print("🎯 MANGUI FI - SYSTÈME AVEC VERROUILLAGE")
        print("=" * 50)
        print("🆕 NOUVEAUTÉS:")
        print("   • Couleurs simplifiées: VERT=Reconnu, ROUGE=Inconnu")
        print("   • Validation: 5 secondes de détection stable requis")
        print("   • Verrouillage: Rectangle fixe pendant 2 minutes après pointage")
        print("   • Analyses masquées: Console plus propre")
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
                        # Détection et reconnaissance de base
                        face_locations, noms = self.detecter_et_reconnaitre(frame)
                        
                        # APPLICATION DE LA NOUVELLE LOGIQUE DE VERROUILLAGE
                        face_locations, noms = self.gerer_verrouillage_et_validation(face_locations, noms)
                        
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
                elif key == ord('v'):
                    self.afficher_statut_verrouillage()
                
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
        """Interface utilisateur avec statut de verrouillage"""
        h, w = frame.shape[:2]
        nb_visages = len(self.derniers_visages)
        
        # En-tête semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Statut références
        statut_ref = f"PERSONNES: {len(self.references_encodings)}/7"
        cv2.putText(frame, statut_ref, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Statut verrouillage
        temps_actuel = time.time()
        if self.personne_verrouillee and temps_actuel < self.temps_fin_verrouillage:
            temps_restant = int(self.temps_fin_verrouillage - temps_actuel)
            statut_verrou = f"VERROUILLÉ: {self.personne_verrouillee} ({temps_restant}s)"
            couleur_verrou = (0, 255, 0)  # Vert
        elif self.validation_nom:
            progression = f"({self.validation_compteur}/{self.validation_requise})"
            statut_verrou = f"VALIDATION: {self.validation_nom} {progression}"
            couleur_verrou = (0, 255, 255)  # Jaune
        else:
            statut_verrou = "EN ATTENTE DE DETECTION"
            couleur_verrou = (255, 255, 255)  # Blanc
        
        cv2.putText(frame, statut_verrou, (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur_verrou, 1)
        
        # Statut principal
        if nb_visages > 0:
            if self.personne_verrouillee:
                statut = f"{self.personne_verrouillee} - POINTÉ ET VERROUILLÉ"
                couleur_statut = (0, 255, 0)
            else:
                statut = f"{nb_visages} VISAGE(S) DÉTECTÉ(S)"
                couleur_statut = (0, 165, 255)
        else:
            statut = "SCANNING..."
            couleur_statut = (255, 255, 255)
        
        cv2.putText(frame, "MANGUI FI - SYSTÈME VERROUILLÉ", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Informations
        info_text = f"Frame: {self.compteur_frames} | Visages: {nb_visages}"
        cv2.putText(frame, info_text, (w - 250, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Pied de page
        cv2.rectangle(frame, (0, h-30), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "Q=Quitter  P=Pointage  S=Stats  L=Liste  V=Statut", (10, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def pointage_manuel(self):
        """Pointage manuel avec verrouillage"""
        if self.derniers_visages:
            try:
                noms_detectes = [nom for nom, _ in self.derniers_noms]
                if noms_detectes and any(nom in self.noms_references for nom in noms_detectes):
                    # Prendre la première personne reconnue
                    for nom, _ in self.derniers_noms:
                        if nom in self.noms_references:
                            self.sauvegarder_pointage(f"{nom} (manuel)", 0.99)
                            
                            # Activer le verrouillage immédiat pour le pointage manuel
                            self.personne_verrouillee = nom
                            self.position_verrouillee = self.derniers_visages[0]
                            self.temps_fin_verrouillage = time.time() + 120  # 2 minutes
                            self.validation_compteur = 0
                            self.validation_nom = None
                            
                            print(f"✅ Pointage manuel et verrouillage pour {nom}")
                            break
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
        print(f"\n👥 LISTE DES PERSONNES ENREGISTRÉES ({len(self.noms_references)}/7):")
        for i, nom in enumerate(self.noms_references, 1):
            print(f"   {i}. {nom}")

    def afficher_statut_verrouillage(self):
        """Affiche le statut actuel du système de verrouillage"""
        print(f"\n🔒 STATUT DU VERROUILLAGE:")
        if self.personne_verrouillee:
            temps_restant = int(self.temps_fin_verrouillage - time.time())
            print(f"   ✅ VERROUILLÉ: {self.personne_verrouillee}")
            print(f"   ⏰ Temps restant: {temps_restant} secondes")
        elif self.validation_nom:
            print(f"   🔄 VALIDATION EN COURS: {self.validation_nom}")
            print(f"   📈 Progression: {self.validation_compteur}/{self.validation_requise}")
        else:
            print(f"   🔓 AUCUN VERROUILLAGE ACTIF")
            print(f"   👀 En attente de détection...")

# Lancement du système
if __name__ == "__main__":
    print("🚀 Démarrage MANGUI FI - Système avec Verrouillage...")
    systeme = SystemeReconnaissanceFaciale()
    systeme.executer()
