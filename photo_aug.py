#!/usr/bin/env python3
"""
MANGUI FI - AUGMENTATION DES DONNÉES
Crée 10 variations du visage d'Alphonse pour améliorer l'entraînement
"""

import cv2
import numpy as np
import os
import pickle
from datetime import datetime

print("🎯 MANGUI FI - AUGMENTATION DES DONNÉES")
print("=" * 50)

class AugmentationDonnees:
    def __init__(self):
        self.chemin_alphonse = "/home/alphonse/facialVCN/VNC_mangui_fi/marie/Alphonse Marie Mbengue.jpg"
        self.modele_embedding = "modele_alphonse_augmente.pkl"
        self.detecteur = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
    def charger_visage_original(self):
        """Charge et détecte le visage original"""
        print("📁 Chargement de l'image originale...")
        image = cv2.imread(self.chemin_alphonse)
        if image is None:
            print(f"❌ Impossible de charger: {self.chemin_alphonse}")
            return None
        
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        visages = self.detecteur.detectMultiScale(gris, 1.1, 5, minSize=(100, 100))
        
        if len(visages) == 0:
            print("❌ Aucun visage détecté")
            return None
        
        x, y, w, h = visages[0]
        visage_original = gris[y:y+h, x:x+w]
        print(f"✅ Visage original détecté: {w}x{h} pixels")
        
        return visage_original
    
    def creer_variations(self, visage_original):
        """Crée 10 variations différentes du visage"""
        print("🎨 Création des variations...")
        variations = []
        
        # 1. Original (normalisé)
        visage_base = self.preprocess_standard(visage_original)
        variations.append(("Original", visage_base))
        
        # 2. Luminosité augmentée (+20%)
        visage_clair = self.ajuster_luminosite(visage_base, 1.2)
        variations.append(("Luminosité +20%", visage_clair))
        
        # 3. Luminosité réduite (-20%)
        visage_sombre = self.ajuster_luminosite(visage_base, 0.8)
        variations.append(("Luminosité -20%", visage_sombre))
        
        # 4. Contraste augmenté
        visage_contraste = self.ajuster_contraste(visage_base, 1.3)
        variations.append(("Contraste +30%", visage_contraste))
        
        # 5. Légère rotation gauche (-5 degrés)
        visage_rotation_gauche = self.rotation_visage(visage_base, -5)
        variations.append(("Rotation -5°", visage_rotation_gauche))
        
        # 6. Légère rotation droite (+5 degrés)
        visage_rotation_droite = self.rotation_visage(visage_base, 5)
        variations.append(("Rotation +5°", visage_rotation_droite))
        
        # 7. Flou gaussien léger
        visage_flou = cv2.GaussianBlur(visage_base, (3, 3), 0)
        variations.append(("Flou léger", visage_flou))
        
        # 8. Zoom léger (105%)
        visage_zoom = self.zoom_visage(visage_base, 1.05)
        variations.append(("Zoom 105%", visage_zoom))
        
        # 9. Dézoom léger (95%)
        visage_dezoom = self.zoom_visage(visage_base, 0.95)
        variations.append(("Zoom 95%", visage_dezoom))
        
        # 10. Bruit gaussien léger
        visage_bruite = self.ajouter_bruit(visage_base)
        variations.append(("Bruit léger", visage_bruite))
        
        # 11. Égalisation d'histogramme renforcée
        visage_egalise = cv2.equalizeHist(visage_base)
        variations.append(("Égalisation renforcée", visage_egalise))
        
        # 12. Miroir horizontal
        visage_miroir = cv2.flip(visage_base, 1)
        variations.append(("Miroir horizontal", visage_miroir))
        
        print(f"✅ {len(variations)} variations créées")
        return variations
    
    def preprocess_standard(self, visage_image):
        """Prétraitement standard pour toutes les variations"""
        visage = cv2.resize(visage_image, (64, 64))
        visage = cv2.equalizeHist(visage)
        return visage
    
    def ajuster_luminosite(self, image, facteur):
        """Ajuste la luminosité de l'image"""
        image_float = image.astype(np.float32)
        image_ajustee = np.clip(image_float * facteur, 0, 255)
        return image_ajustee.astype(np.uint8)
    
    def ajuster_contraste(self, image, facteur):
        """Ajuste le contraste de l'image"""
        image_float = image.astype(np.float32)
        moyenne = np.mean(image_float)
        image_ajustee = np.clip((image_float - moyenne) * facteur + moyenne, 0, 255)
        return image_ajustee.astype(np.uint8)
    
    def rotation_visage(self, image, angle):
        """Effectue une rotation du visage"""
        h, w = image.shape
        centre = (w // 2, h // 2)
        matrice_rotation = cv2.getRotationMatrix2D(centre, angle, 1.0)
        image_rotation = cv2.warpAffine(image, matrice_rotation, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return image_rotation
    
    def zoom_visage(self, image, facteur_zoom):
        """Effectue un zoom sur le visage"""
        h, w = image.shape
        nouvelle_taille = int(min(h, w) * facteur_zoom)
        debut_h = (h - nouvelle_taille) // 2
        debut_w = (w - nouvelle_taille) // 2
        
        if facteur_zoom > 1:  # Zoom
            image_zoom = image[debut_h:debut_h+nouvelle_taille, debut_w:debut_w+nouvelle_taille]
            image_zoom = cv2.resize(image_zoom, (w, h))
        else:  # Dézoom
            image_redimensionnee = cv2.resize(image, (nouvelle_taille, nouvelle_taille))
            image_zoom = np.zeros((h, w), dtype=np.uint8)
            debut_h_final = (h - nouvelle_taille) // 2
            debut_w_final = (w - nouvelle_taille) // 2
            image_zoom[debut_h_final:debut_h_final+nouvelle_taille, debut_w_final:debut_w_final+nouvelle_taille] = image_redimensionnee
        
        return image_zoom
    
    def ajouter_bruit(self, image):
        """Ajoute un bruit gaussien léger"""
        bruit = np.random.normal(0, 5, image.shape).astype(np.float32)
        image_bruite = image.astype(np.float32) + bruit
        return np.clip(image_bruite, 0, 255).astype(np.uint8)
    
    def creer_modele_moyen(self, variations):
        """Crée un modèle basé sur la moyenne de toutes les variations"""
        print("�� Création du modèle moyen...")
        
        # Extraire tous les embeddings
        embeddings = []
        for nom, visage in variations:
            embedding = {
                'pixels': visage.flatten(),
                'nom_variation': nom
            }
            embeddings.append(embedding)
        
        # Calculer l'embedding moyen (plus robuste)
        pixels_moyens = np.mean([emb['pixels'] for emb in embeddings], axis=0)
        
        # Créer le modèle final
        modele_final = {
            'pixels': pixels_moyens.astype(np.float32),
            'taille_visage': (64, 64),
            'nombre_variations': len(variations),
            'timestamp': datetime.now().isoformat(),
            'seuil_recommandé': 0.65,  # Seuil plus bas car modèle plus robuste
            'version_modele': '3.0_augmente',
            'variations_incluses': [nom for nom, _ in variations]
        }
        
        return modele_final
    
    def sauvegarder_variations_images(self, variations):
        """Sauvegarde les variations en images pour vérification"""
        dossier_variations = "variations_alphonse"
        os.makedirs(dossier_variations, exist_ok=True)
        
        for i, (nom, visage) in enumerate(variations):
            chemin_image = f"{dossier_variations}/variation_{i+1:02d}_{nom}.jpg"
            cv2.imwrite(chemin_image, visage)
        
        print(f"📷 Variations sauvegardées dans: {dossier_variations}/")
    
    def executer_augmentation(self):
        """Exécute tout le processus d'augmentation"""
        print("🚀 LANCEMENT DE L'AUGMENTATION...")
        
        # 1. Charger le visage original
        visage_original = self.charger_visage_original()
        if visage_original is None:
            return False
        
        # 2. Créer les variations
        variations = self.creer_variations(visage_original)
        
        # 3. Sauvegarder les variations en images (optionnel)
        self.sauvegarder_variations_images(variations)
        
        # 4. Créer le modèle moyen
        modele_final = self.creer_modele_moyen(variations)
        
        # 5. Sauvegarder le modèle
        with open(self.modele_embedding, 'wb') as f:
            pickle.dump(modele_final, f)
        
        print(f"\n✅ MODÈLE AUGMENTÉ SAUVEGARDÉ: {self.modele_embedding}")
        print(f"📊 Statistiques:")
        print(f"   - Variations: {modele_final['nombre_variations']}")
        print(f"   - Dimensions: {modele_final['taille_visage']}")
        print(f"   - Seuil recommandé: {modele_final['seuil_recommandé']}")
        print(f"   - Plage pixels: {np.min(modele_final['pixels']):.1f} à {np.max(modele_final['pixels']):.1f}")
        
        return True

if __name__ == "__main__":
    augmentation = AugmentationDonnees()
    
    if augmentation.executer_augmentation():
        print("\n🎉 AUGMENTATION RÉUSSIE!")
        print("💡 Maintenant exécutez: python3 reconnaissance_augmentee.py")
    else:
        print("\n❌ Échec de l'augmentation")
