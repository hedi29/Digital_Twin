import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

class DigitalTwin:
    def __init__(self, persona_id, interaction_history, demographics):
        self.persona_id = persona_id
        self.interaction_history = interaction_history
        self.demographics = demographics
        self.preferences = self._extract_preferences()
    
    def _extract_preferences(self):
        """Extract preference patterns from interaction history"""
        # Categories with highest engagement time
        category_engagement = self.interaction_history.groupby('category')['time_spent'].mean()
        top_categories = category_engagement.nlargest(5).index.tolist()
        
        # Format preferences
        format_pref = self.interaction_history.groupby('format')['time_spent'].mean()
        
        return {
            'top_categories': top_categories,
            'format_preference': format_pref.to_dict(),
            'avg_engagement_time': self.interaction_history['time_spent'].mean()
        }
    
    def evaluate_concept(self, concept):
        """Evaluate a new marketing concept"""
        score = 0
        reasons = []
        
        # Category match scoring
        category_matches = len(set(concept['categories']) & set(self.preferences['top_categories']))
        score += category_matches * 20
        
        # Format preference scoring
        format_score = self.preferences['format_preference'].get(concept['format'], 0)
        normalized_format_score = min(format_score / self.preferences['avg_engagement_time'], 2) * 30
        score += normalized_format_score
        
        # Demographic alignment
        if self.demographics['age_group'] == concept.get('target_age'):
            score += 25
            reasons.append(f"Aligns with {self.demographics['age_group']} age preferences")
        
        # Generate response
        if score >= 60:
            response = "yes"
            reasons.append("Strong alignment with past engagement patterns")
        elif score >= 40:
            response = "maybe"
            reasons.append("Moderate interest based on partial category match")
        else:
            response = "no"
            reasons.append("Low relevance to demonstrated interests")
        
        return {
            'response': response,
            'score': score,
            'reasons': ' '.join(reasons)
        }

class PersonaFactory:
    def __init__(self, user_interactions_df):
        self.interactions = user_interactions_df
        self.personas = {}
        
    def create_personas(self, n_clusters=5):
        """Create personas using K-means clustering"""
        # Create user-feature matrix
        user_features = self.interactions.pivot_table(
            index='user_id',
            columns='category',
            values='time_spent',
            aggfunc='mean',
            fill_value=0
        )
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(user_features)
        
        # Cluster users
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Create digital twins for each cluster
        for cluster_id in range(n_clusters):
            cluster_users = user_features.index[clusters == cluster_id]
            cluster_interactions = self.interactions[self.interactions['user_id'].isin(cluster_users)]
            
            # Mock demographics for demo
            demographics = {
                'age_group': ['18-24', '25-34', '35-44', '45-54', '55+'][cluster_id],
                'cluster_size': len(cluster_users)
            }
            
            self.personas[f'persona_{cluster_id}'] = DigitalTwin(
                persona_id=f'persona_{cluster_id}',
                interaction_history=cluster_interactions,
                demographics=demographics
            )
        
        return self.personas

# Example usage
if __name__ == "__main__":
    # Simulate interaction data
    np.random.seed(42)
    interactions = []
    
    for user_id in range(100):
        for _ in range(20):  # 20 interactions per user
            interactions.append({
                'user_id': user_id,
                'category': np.random.choice(['tech', 'fashion', 'food', 'travel', 'fitness']),
                'format': np.random.choice(['image', 'video', 'reel']),
                'time_spent': np.random.exponential(scale=10)
            })
    
    interactions_df = pd.DataFrame(interactions)
    
    # Create personas
    factory = PersonaFactory(interactions_df)
    personas = factory.create_personas()
    
    # Test with a new concept
    new_concept = {
        'categories': ['tech', 'fitness'],
        'format': 'video',
        'target_age': '25-34',
        'description': 'Smart fitness tracker ad with tech focus'
    }
    
    # Evaluate concept across all personas
    print("Concept Evaluation Results:")
    print("-" * 50)
    for persona_name, twin in personas.items():
        result = twin.evaluate_concept(new_concept)
        print(f"{persona_name}: {result['response'].upper()} (score: {result['score']})")
        print(f"Reason: {result['reasons']}\n")