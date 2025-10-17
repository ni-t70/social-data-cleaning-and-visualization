# Step 1: Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("Libraries imported successfully!")


# Step 2: Generate Simulated Social Media Data

def generate_social_media_data(n_posts=1000):
    """
    Generate simulated social media data for analysis
    """
    random.seed(42)
    np.random.seed(42)
    
    # Define categories
    categories = ['Technology', 'Sports', 'Entertainment', 'News', 'Food', 
                  'Travel', 'Fashion', 'Health', 'Business', 'Education']
    
    # Generate random data
    data = {
        'post_id': range(1, n_posts + 1),
        'category': [random.choice(categories) for _ in range(n_posts)],
        'likes': np.random.randint(0, 10000, n_posts),
        'shares': np.random.randint(0, 5000, n_posts),
        'comments': np.random.randint(0, 1000, n_posts),
        'timestamp': [datetime.now() - timedelta(days=random.randint(0, 365)) 
                      for _ in range(n_posts)],
        'user_followers': np.random.randint(100, 100000, n_posts)
    }
    
    # Introduce some missing values intentionally
    df = pd.DataFrame(data)
    missing_indices = random.sample(range(n_posts), k=int(n_posts * 0.05))
    df.loc[missing_indices, 'likes'] = np.nan
    
    return df

# Generate data
df = generate_social_media_data(1000)
print(f"\nDataset created with {len(df)} posts")
print(df.head())


# Step 3: Data Exploration

print("\n" + "="*70)
print("DATA EXPLORATION")
print("="*70)

# Basic information
print("\nDataset Info:")
print(df.info())

print("\nDataset Shape:", df.shape)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\nFirst 5 rows:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCategory Distribution:")
print(df['category'].value_counts())

# Step 4: Data Cleaning

print("\n" + "="*70)
print("DATA CLEANING")
print("="*70)

# Create a copy for cleaning
df_clean = df.copy()

# Handle missing values in likes column
print(f"\nMissing likes before cleaning: {df_clean['likes'].isnull().sum()}")

# Fill missing likes with median value by category
df_clean['likes'] = df_clean.groupby('category')['likes'].transform(
    lambda x: x.fillna(x.median())
)

print(f"Missing likes after cleaning: {df_clean['likes'].isnull().sum()}")

# Remove any remaining rows with missing values
df_clean = df_clean.dropna()

# Remove duplicates
duplicates = df_clean.duplicated().sum()
print(f"\nDuplicates found: {duplicates}")
df_clean = df_clean.drop_duplicates()

# Convert data types
df_clean['likes'] = df_clean['likes'].astype(int)
df_clean['shares'] = df_clean['shares'].astype(int)
df_clean['comments'] = df_clean['comments'].astype(int)

# Create engagement score
df_clean['engagement_score'] = (
    df_clean['likes'] + 
    df_clean['shares'] * 2 + 
    df_clean['comments'] * 3
)

print(f"\nCleaned dataset shape: {df_clean.shape}")
print("\nCleaned data sample:")
print(df_clean.head())


# Step 5: Data Analysis

print("\n" + "="*70)
print("DATA ANALYSIS")
print("="*70)

# Analysis by category
category_stats = df_clean.groupby('category').agg({
    'likes': ['mean', 'median', 'sum', 'count'],
    'shares': 'mean',
    'comments': 'mean',
    'engagement_score': 'mean'
}).round(2)

print("\nStatistics by Category:")
print(category_stats)

# Top categories by engagement
top_categories = df_clean.groupby('category')['engagement_score'].mean().sort_values(ascending=False)
print("\nTop Categories by Engagement Score:")
print(top_categories)

# Correlation analysis
print("\nCorrelation Matrix:")
correlation = df_clean[['likes', 'shares', 'comments', 'user_followers', 'engagement_score']].corr()
print(correlation)


# Step 6: Data Visualization

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Figure 1: Distribution of Likes
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df_clean['likes'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Number of Likes')
plt.ylabel('Frequency')
plt.title('Distribution of Likes')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df_clean['likes'])
plt.ylabel('Number of Likes')
plt.title('Boxplot of Likes')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Figure 2: Likes by Category
plt.figure(figsize=(14, 6))

category_likes = df_clean.groupby('category')['likes'].mean().sort_values(ascending=False)

plt.subplot(1, 2, 1)
category_likes.plot(kind='bar', color='coral')
plt.xlabel('Category')
plt.ylabel('Average Likes')
plt.title('Average Likes by Category')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.pie(df_clean.groupby('category').size(), labels=category_likes.index, 
        autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Posts by Category')

plt.tight_layout()
plt.show()

# Figure 3: Engagement Metrics Comparison
plt.figure(figsize=(14, 6))

engagement_by_category = df_clean.groupby('category')[['likes', 'shares', 'comments']].mean()

engagement_by_category.plot(kind='bar', width=0.8)
plt.xlabel('Category')
plt.ylabel('Average Count')
plt.title('Average Engagement Metrics by Category')
plt.legend(['Likes', 'Shares', 'Comments'])
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 4: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap of Engagement Metrics')
plt.tight_layout()
plt.show()

# Figure 5: Engagement Score by Category
plt.figure(figsize=(12, 6))

engagement_scores = df_clean.groupby('category')['engagement_score'].mean().sort_values(ascending=False)

plt.barh(engagement_scores.index, engagement_scores.values, color='mediumseagreen')
plt.xlabel('Average Engagement Score')
plt.ylabel('Category')
plt.title('Engagement Score by Category')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 6: Scatter plot - Followers vs Likes
plt.figure(figsize=(10, 6))
plt.scatter(df_clean['user_followers'], df_clean['likes'], 
            alpha=0.5, c=df_clean['engagement_score'], cmap='viridis')
plt.xlabel('User Followers')
plt.ylabel('Likes')
plt.title('Relationship between User Followers and Likes')
plt.colorbar(label='Engagement Score')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Step 7: Key Insights & Conclusions

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

# Recalculate all metrics to ensure they're available
if 'top_categories' not in locals():
    top_categories = df_clean.groupby('category')['engagement_score'].mean().sort_values(ascending=False)

if 'correlation' not in locals():
    correlation = df_clean[['likes', 'shares', 'comments', 'user_followers', 'engagement_score']].corr()

# Calculate insights
top_category = top_categories.index[0]
top_engagement = top_categories.values[0]
total_likes = df_clean['likes'].sum()
avg_likes = df_clean['likes'].mean()
median_likes = df_clean['likes'].median()
likes_shares_corr = correlation.loc['likes', 'shares']
likes_comments_corr = correlation.loc['likes', 'comments']
followers_likes_corr = correlation.loc['user_followers', 'likes']
total_categories = df_clean['category'].nunique()
most_frequent_category = df_clean['category'].mode()[0]
total_posts = len(df_clean)
missing_handled = df['likes'].isnull().sum()
data_completeness = (1 - df_clean.isnull().sum().sum() / df_clean.size) * 100

print(f"""
1. MOST POPULAR CATEGORY:
   - {top_category} has the highest average engagement score: {top_engagement:.2f}

2. ENGAGEMENT METRICS:
   - Total likes across all posts: {total_likes:,}
   - Average likes per post: {avg_likes:.2f}
   - Median likes per post: {median_likes:.2f}

3. CORRELATION FINDINGS:
   - Likes and Shares correlation: {likes_shares_corr:.3f}
   - Likes and Comments correlation: {likes_comments_corr:.3f}
   - Followers and Likes correlation: {followers_likes_corr:.3f}

4. CATEGORY DISTRIBUTION:
   - Total categories analyzed: {total_categories}
   - Most frequent category: {most_frequent_category}

5. DATA QUALITY:
   - Posts analyzed: {total_posts:,}
   - Missing values handled: {missing_handled}
   - Data completeness: {data_completeness:.2f}%
""")

print("="*70)
print("ANALYSIS COMPLETE!")
print("="*70)


# Conclusion

# The social media data cleaning and analysis project successfully simulated and explored user engagement patterns across various content categories. Through systematic data generation, cleaning, and visualization, valuable insights were obtained regarding audience interaction on a hypothetical platform.
# The analysis revealed that Fashion posts achieved the highest average engagement score, suggesting that visually appealing and lifestyle oriented content tends to perform better in terms of likes, shares, and comments. Other categories such as Travel and Business also showed strong engagement, indicating that aspirational and informative content drives significant user interaction.
# From the correlation findings, it was observed that the relationships between likes, shares, and comments were relatively weak, implying that each engagement metric may represent distinct user behaviors. Interestingly, the correlation between followers and likes was slightly negative, suggesting that follower count alone does not guarantee higher engagement content quality and relevance may play more crucial roles.
# The data cleaning process ensured high data integrity, with all missing values handled effectively and no duplicate records remaining. The dataset achieved 100% completeness, ensuring reliability in subsequent analysis.
# Overall, this project demonstrates how simulated social media data can be leveraged to uncover trends in user engagement and category performance. The insights can guide businesses and content creators in optimizing their social media strategies — focusing on categories that foster higher engagement, crafting shareable content, and understanding the diverse factors that drive audience interaction online.
