import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import csv
import os
from datetime import datetime


class MovieRecommender:
    def __init__(self, root):
        # Initialize the main application window
        self.root = root
        self.root.title("Movie System")
        self.root.geometry("1100x650")  # Set window size

        # Initialize data storage variables (will be loaded from CSV files)
        self.movies = None  # movie data (titles, genres)
        self.ratings = None  # user ratings data
        self.tfidf_matrix = None  # numerical representation of movie genres
        self.knn_model = None  # model for finding similar movies
        self.user_matrix = None  # User-movie rating matrix for collaborative filtering
        self.movie_avg_ratings = None  # Average rating for each movie

        # File paths for data (CHANGE THESE TO YOUR ACTUAL FILE LOCATIONS)
        self.movies_path = r"C:\Users\COM-A\Downloads\Dataset MOVIE RECOMMENDER\movies.csv"
        self.ratings_path = r"C:\Users\COM-A\Downloads\Dataset MOVIE RECOMMENDER\ratings.csv"
        self.feedback_file = "feedback.csv"  # Local file to store user feedback

        # Create the user interface and load data
        self.create_interface()
        self.load_data()

    def create_interface(self):
        # Create main title label
        tk.Label(self.root, text="MOVIE SYSTEM", font=("Arial", 16, "bold"), fg="Grey").pack(pady=10)

        # Create notebook (tabbed interface)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create all 5 tabs
        self.create_content_tab(notebook)
        self.create_collab_tab(notebook)
        self.create_hybrid_tab(notebook)
        self.create_search_tab(notebook)
        self.create_feedback_tab(notebook)

        # Create status bar at the bottom
        self.status = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def create_content_tab(self, notebook):
        """Create Tab 1: Content-based filtering based on movie genres"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Content-Based')

        # Input section for movie title(s)
        tk.Label(frame, text="Movie Title:").pack(anchor=tk.W, padx=10, pady=(10, 5))
        tk.Label(frame, text="(Separate with spaces)").pack(anchor=tk.W, padx=10, pady=(0, 0))
        self.movie_input = tk.StringVar()  # Variable to store user input
        tk.Entry(frame, textvariable=self.movie_input, width=40).pack(padx=10, pady=(0, 10))

        # Input for number of recommendations
        tk.Label(frame, text="Number of Recommendations:").pack(anchor=tk.W, padx=10)
        self.num_recs = tk.StringVar(value="10")  # Default value is 10
        ttk.Spinbox(frame, from_=5, to=30, textvariable=self.num_recs, width=10).pack(padx=10, pady=(0, 10))

        # Button to trigger content-based recommendations
        tk.Button(frame, text="Find Similar Movies", command=self.get_content_recs, bg="lightblue").pack(padx=10,
                                                                                                         pady=(0, 10))

        # Text area to display results
        self.content_results = scrolledtext.ScrolledText(frame, height=18)
        self.content_results.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def create_collab_tab(self, notebook):
        """Create Tab 2: Collaborative filtering based on user ratings"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Collaborative')

        # Input section for user ID(s)
        tk.Label(frame, text="User ID(s):").pack(anchor=tk.W, padx=10, pady=(10, 5))
        tk.Label(frame, text="(Separate with spaces)").pack(anchor=tk.W, padx=10, pady=(0, 0))
        self.user_input = tk.StringVar()
        tk.Entry(frame, textvariable=self.user_input, width=40).pack(padx=10, pady=(0, 10))

        # Input for number of recommendations
        tk.Label(frame, text="Number of Recommendations:").pack(anchor=tk.W, padx=10)
        self.collab_num = tk.StringVar(value="10")
        ttk.Spinbox(frame, from_=5, to=30, textvariable=self.collab_num, width=10).pack(padx=10, pady=(0, 10))

        # Button to trigger collaborative recommendations
        tk.Button(frame, text="Get Recommendations", command=self.get_collab_recs, bg="lightgreen").pack(padx=10,
                                                                                                         pady=(0, 10))

        # Text area to display results
        self.collab_results = scrolledtext.ScrolledText(frame, height=18)
        self.collab_results.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def create_hybrid_tab(self, notebook):
        """Create Tab 3: Hybrid approach combining both methods"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Hybrid')

        # Input for movie title(s) - content-based part
        tk.Label(frame, text="Movie Title(s):").pack(anchor=tk.W, padx=10, pady=(10, 5))
        tk.Label(frame, text="(Separate with spaces)").pack(anchor=tk.W, padx=10, pady=(0, 0))
        self.hybrid_movie = tk.StringVar()
        tk.Entry(frame, textvariable=self.hybrid_movie, width=30).pack(padx=10, pady=(0, 10))

        # Input for user ID(s) - collaborative part (optional)
        tk.Label(frame, text="User ID(s) (Optional):").pack(anchor=tk.W, padx=10)
        tk.Label(frame, text="(Separate with spaces)").pack(anchor=tk.W, padx=10, pady=(0, 0))
        self.hybrid_user = tk.StringVar()
        tk.Entry(frame, textvariable=self.hybrid_user, width=30).pack(padx=10, pady=(0, 10))

        # Input for total number of recommendations
        tk.Label(frame, text="Number:").pack(anchor=tk.W, padx=10)
        self.hybrid_num = tk.StringVar(value="12")
        ttk.Spinbox(frame, from_=5, to=30, textvariable=self.hybrid_num, width=8).pack(padx=10, pady=(0, 10))

        # Button to trigger hybrid recommendations
        tk.Button(frame, text="Get Hybrid Recommendations", command=self.get_hybrid_recs, bg="orange").pack(padx=10,
                                                                                                            pady=(
                                                                                                            0, 10))

        # Text area to display results
        self.hybrid_results = scrolledtext.ScrolledText(frame, height=18)
        self.hybrid_results.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def create_search_tab(self, notebook):
        """Create Tab 4: Search movies by title"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Search')

        # Input for search term(s)
        tk.Label(frame, text="Search Term(s):").pack(anchor=tk.W, padx=10, pady=(10, 5))
        tk.Label(frame, text="(Separate with spaces)").pack(anchor=tk.W, padx=10, pady=(0, 0))
        self.search_input = tk.StringVar()
        tk.Entry(frame, textvariable=self.search_input, width=40).pack(padx=10, pady=(0, 10))

        # Button to search movies
        tk.Button(frame, text="Search", command=self.search_movies).pack(padx=10, pady=(0, 10))

        # Text area to display search results
        self.search_results = scrolledtext.ScrolledText(frame, height=18)
        self.search_results.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def create_feedback_tab(self, notebook):
        """Create Tab 5: User feedback system"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Feedback')

        # Left side: Feedback form
        form_frame = tk.Frame(frame)
        form_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Movie title input for feedback
        tk.Label(form_frame, text="Movie Title:").pack(anchor=tk.W, pady=(0, 5))
        self.fb_movie = tk.StringVar()
        tk.Entry(form_frame, textvariable=self.fb_movie, width=30).pack(fill=tk.X, pady=(0, 10))

        # Rating input (1-5 stars)
        tk.Label(form_frame, text="Rating (1-5):").pack(anchor=tk.W, pady=(0, 5))
        self.fb_rating = tk.IntVar(value=3)  # Default rating is 3
        rating_frame = tk.Frame(form_frame)
        rating_frame.pack(fill=tk.X, pady=(0, 10))
        for i in range(1, 6):
            tk.Radiobutton(rating_frame, text=str(i), variable=self.fb_rating, value=i).pack(side=tk.LEFT, padx=2)

        # Comments input
        tk.Label(form_frame, text="Comments:").pack(anchor=tk.W, pady=(0, 5))
        self.fb_comments = tk.Text(form_frame, height=3)
        self.fb_comments.pack(fill=tk.X, pady=(0, 10))

        # Submit feedback button
        tk.Button(form_frame, text="Submit Feedback", command=self.submit_feedback, bg="green", fg="white").pack(
            pady=10)

        # Right side: Feedback history
        history_frame = tk.Frame(frame)
        history_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.fb_history = scrolledtext.ScrolledText(history_frame, height=18)
        self.fb_history.pack(fill=tk.BOTH, expand=True)

        # Load existing feedback when tab is created
        self.load_feedback()

    def load_data(self):
        """Load and process data from CSV files"""
        try:
            self.status.config(text="Loading data...")

            # PART 1: LOAD AND PROCESS MOVIES.CSV
            if os.path.exists(self.movies_path):
                # Load movies data using pandas
                self.movies = pd.read_csv(self.movies_path)

                # Clean and format genres column:
                # 1. Fill missing values with empty string
                # 2. Convert to string type
                # 3. Replace pipe separator '|' with space ' ' for better processing
                # Example: "Action|Adventure|Comedy" becomes "Action Adventure Comedy"
                self.movies['genres'] = self.movies['genres'].fillna('').astype(str).str.replace('|', ' ')

                # Create features column for machine learning (same as genres)
                self.movies['features'] = self.movies['genres']

                # Create TF-IDF matrix (converts text genres to numerical vectors)
                # TF-IDF = Term Frequency-Inverse Document Frequency
                # This converts text data into numbers that computers can understand
                tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
                self.tfidf_matrix = tfidf.fit_transform(self.movies['features'])

                # Build K-Nearest Neighbors model for finding similar movies
                # metric='cosine' uses cosine similarity (measures angle between vectors)
                # Smaller angle = more similar movies
                self.knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
                self.knn_model.fit(self.tfidf_matrix)  # Train the model

                messagebox.showinfo("Success", f"Loaded {len(self.movies)} movies!")
            else:
                messagebox.showerror("Error", f"movies.csv not found")
                return

            # PART 2: LOAD AND PROCESS RATINGS.CSV
            if os.path.exists(self.ratings_path):
                # Load ratings data
                self.ratings = pd.read_csv(self.ratings_path)

                # Limit to first 50,000 ratings for better performance
                ratings_sample = self.ratings.head(50000)

                # Create user-item matrix (pivot table):
                # Rows: User IDs
                # Columns: Movie IDs
                # Values: Ratings (0-5), 0 means not rated
                self.user_matrix = ratings_sample.pivot_table(
                    index='userId',
                    columns='movieId',
                    values='rating',
                    fill_value=0
                )

                # Calculate average rating for each movie
                self.movie_avg_ratings = self.ratings.groupby('movieId')['rating'].mean()

                messagebox.showinfo("Success", f"Loaded {len(self.ratings)} ratings!")
            else:
                # If ratings.csv not found, create sample data for testing
                messagebox.showwarning("Warning", "ratings.csv not found. Using sample data.")
                self.create_sample_ratings()

            # Update status bar with loaded data info
            self.status.config(text=f"Ready! Movies: {len(self.movies)}")

        except Exception as e:
            # Show error message if something goes wrong
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status.config(text="Error loading data")

    def create_sample_ratings(self):
        """Create sample rating data if ratings.csv is not available"""
        # Get first 50 movie IDs
        movie_ids = self.movies['movieId'].head(50).tolist()
        sample_data = []

        # Create 10 sample users
        for user_id in range(1, 11):
            # Each user rates 15 random movies
            for movie_id in movie_ids[:15]:
                # Generate random rating between 1-5
                sample_data.append({'userId': user_id, 'movieId': movie_id, 'rating': np.random.randint(1, 6)})

        # Create DataFrame from sample data
        self.ratings = pd.DataFrame(sample_data)

        # Create user-item matrix from sample data
        self.user_matrix = self.ratings.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )

        # Calculate average ratings for sample data
        self.movie_avg_ratings = self.ratings.groupby('movieId')['rating'].mean()

    def get_movie_rating(self, movie_id):
        """Get average rating for a movie, or return 'N/A' if no ratings available"""
        try:
            # Check if movie has ratings and return formatted average
            if self.movie_avg_ratings is not None and movie_id in self.movie_avg_ratings.index:
                return f"{self.movie_avg_ratings[movie_id]:.2f}/5"
            return "N/A"  # No rating available
        except:
            return "N/A"  # Error occurred

    def get_content_recs(self):
        """CONTENT-BASED FILTERING: Find movies similar to input movie(s) based on genres"""
        if self.movies is None:
            return messagebox.showwarning("Warning", "Please load movies data first")

        movie_input = self.movie_input.get().strip()
        if not movie_input:
            return messagebox.showwarning("Warning", "Please enter movie title(s)")

        try:
            # Get number of recommendations requested
            n = int(self.num_recs.get())

            # Clear previous results
            self.content_results.delete('1.0', tk.END)

            # Split input by spaces to handle multiple search terms
            search_terms = movie_input.split()
            all_recommendations = {}  # Dictionary to store combined recommendations

            # For each search term entered by user
            for term in search_terms:
                # Find movies containing the search term (case-insensitive)
                matches = self.movies[self.movies['title'].str.contains(term, case=False, na=False)]
                if len(matches) == 0:
                    continue  # Skip if no matches found

                # Get the first matching movie
                movie_idx = matches.index[0]

                # Use KNN model to find similar movies
                # n_neighbors=n+1: Get n similar movies + the movie itself
                distances, indices = self.knn_model.kneighbors(
                    self.tfidf_matrix[movie_idx],
                    n_neighbors=n + 1
                )

                # Process each similar movie found
                for dist, idx in zip(distances[0][1:], indices[0][1:]):
                    movie = self.movies.iloc[idx]
                    similarity = 1 - dist  # Convert distance to similarity score (0-1)

                    # Store recommendation, keeping the highest similarity score
                    if movie['title'] not in all_recommendations or similarity > all_recommendations[movie['title']][
                        'similarity']:
                        all_recommendations[movie['title']] = {
                            'similarity': similarity,
                            'genres': movie['genres'],
                            'movie_id': movie['movieId']
                        }

            # If no recommendations found, show error message
            if not all_recommendations:
                self.content_results.insert(tk.END, "No movies found.\nTry: ")
                for title in self.movies['title'].head(5):
                    self.content_results.insert(tk.END, f"{title}, ")
                return

            # Sort recommendations by similarity score (highest first)
            sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1]['similarity'], reverse=True)

            # Display results header
            self.content_results.insert(tk.END, f"Movies similar to: {', '.join(search_terms)}\n")
            self.content_results.insert(tk.END, "=" * 50 + "\n\n")

            # Display each recommendation with details
            for i, (title, data) in enumerate(sorted_recs[:n], 1):
                self.content_results.insert(tk.END, f"{i}. {title}\n")
                self.content_results.insert(tk.END, f"   Similarity: {data['similarity']:.3f}\n")
                self.content_results.insert(tk.END, f"   Rating: {self.get_movie_rating(data['movie_id'])}\n")
                self.content_results.insert(tk.END, f"   Genres: {data['genres'][:60]}\n\n")

            # Update status bar
            self.status.config(text=f"Found {min(n, len(sorted_recs))} recommendations")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def get_collab_recs(self):
        """COLLABORATIVE FILTERING: Recommend movies based on user ratings"""
        if self.user_matrix is None:
            return messagebox.showwarning("Warning", "No rating data available")

        try:
            user_input_str = self.user_input.get().strip()
            if not user_input_str:
                return messagebox.showwarning("Warning", "Please enter user ID(s)")

            n = int(self.collab_num.get())
            self.collab_results.delete('1.0', tk.END)
            user_terms = user_input_str.split()
            all_recommendations = {}

            # Process each user ID entered
            for user_term in user_terms:
                try:
                    user_id = int(user_term)

                    # Check if user exists in dataset
                    if user_id not in self.user_matrix.index:
                        continue  # Skip if user not found

                    # Get user's ratings
                    user_ratings = self.user_matrix.loc[user_id]

                    # Find movies the user has already rated (rating > 0)
                    rated_movies = set(user_ratings[user_ratings > 0].index)

                    # Calculate average ratings for all movies
                    avg_ratings = self.user_matrix.mean()

                    # Get movies user hasn't rated yet
                    unrated_movies = avg_ratings[~avg_ratings.index.isin(rated_movies)]

                    # Get top movies with highest average ratings
                    top_movies = unrated_movies.sort_values(ascending=False).head(n * 2)

                    # Process each recommended movie
                    for movie_id, avg_rating in top_movies.items():
                        movie_info = self.movies[self.movies['movieId'] == movie_id]
                        if not movie_info.empty:
                            movie = movie_info.iloc[0]

                            # Store recommendation, keeping highest average rating
                            if movie['title'] not in all_recommendations or avg_rating > \
                                    all_recommendations[movie['title']]['avg_rating']:
                                all_recommendations[movie['title']] = {
                                    'avg_rating': avg_rating,
                                    'genres': movie['genres'],
                                    'movie_id': movie_id,
                                    'user': user_id
                                }

                except ValueError:
                    continue  # Skip invalid user IDs

            # If no recommendations found, show error
            if not all_recommendations:
                self.collab_results.insert(tk.END, "No valid users found.\nTry: ")
                for uid in list(self.user_matrix.index)[:5]:
                    self.collab_results.insert(tk.END, f"{uid}, ")
                return

            # Sort recommendations by average rating (highest first)
            sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1]['avg_rating'], reverse=True)

            # Display results header
            self.collab_results.insert(tk.END, f"Recommendations for user(s): {', '.join(user_terms)}\n")
            self.collab_results.insert(tk.END, "=" * 50 + "\n\n")

            # Display each recommendation
            for i, (movie_title, data) in enumerate(sorted_recs[:n], 1):
                self.collab_results.insert(tk.END, f"{i}. {movie_title}\n")
                self.collab_results.insert(tk.END, f"   Avg Rating: {data['avg_rating']:.2f}/5\n")
                self.collab_results.insert(tk.END, f"   User Rating: {self.get_movie_rating(data['movie_id'])}\n")
                self.collab_results.insert(tk.END, f"   Genres: {data['genres'][:60]}\n")

                # If searching for multiple users, show which user this is for
                if len(user_terms) > 1:
                    self.collab_results.insert(tk.END, f"   For User: {data['user']}\n")
                self.collab_results.insert(tk.END, "\n")

            self.status.config(text=f"Found {min(n, len(sorted_recs))} recommendations")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def get_hybrid_recs(self):
        """HYBRID RECOMMENDATIONS: Combine content-based and collaborative filtering"""
        if self.movies is None:
            return messagebox.showwarning("Warning", "Please load data first")

        # Clear previous results and display header
        self.hybrid_results.delete('1.0', tk.END)
        self.hybrid_results.insert(tk.END, "HYBRID RECOMMENDATIONS\n")
        self.hybrid_results.insert(tk.END, "=" * 50 + "\n\n")

        recommendations = []  # List to store all recommendations

        # PART 1: CONTENT-BASED FILTERING (from multiple movies)
        movie_input = self.hybrid_movie.get().strip()
        if movie_input:
            search_terms = movie_input.split()
            content_recs = {}  # Dictionary for content-based recommendations

            # For each movie search term
            for term in search_terms:
                matches = self.movies[self.movies['title'].str.contains(term, case=False, na=False)]
                if len(matches) > 0:
                    movie_idx = matches.index[0]
                    n = int(self.hybrid_num.get()) // 2  # Half of total for content-based

                    # Find similar movies using KNN
                    distances, indices = self.knn_model.kneighbors(
                        self.tfidf_matrix[movie_idx],
                        n_neighbors=n + 1
                    )

                    # Store content-based recommendations
                    for dist, idx in zip(distances[0][1:], indices[0][1:]):
                        movie = self.movies.iloc[idx]
                        if movie['title'] not in content_recs:
                            content_recs[movie['title']] = 1 - dist  # Store similarity score

            # Display content-based recommendations
            if content_recs:
                self.hybrid_results.insert(tk.END, f"From {len(search_terms)} movie(s): {', '.join(search_terms)}\n")
                self.hybrid_results.insert(tk.END, "-" * 30 + "\n")

                # Sort by similarity and display
                sorted_content = sorted(content_recs.items(), key=lambda x: x[1], reverse=True)
                for i, (title, similarity) in enumerate(sorted_content[:int(self.hybrid_num.get()) // 2], 1):
                    if title not in recommendations:
                        recommendations.append(title)
                        self.hybrid_results.insert(tk.END, f"{i}. {title}\n")
                        self.hybrid_results.insert(tk.END, f"   [Content] Similarity: {similarity:.3f}\n\n")

        # PART 2: COLLABORATIVE FILTERING (from multiple users)
        user_str = self.hybrid_user.get().strip()
        if user_str and self.user_matrix is not None:
            user_terms = user_str.split()
            collab_recs = {}  # Dictionary for collaborative recommendations

            # For each user ID
            for user_term in user_terms:
                try:
                    user_id = int(user_term)
                    if user_id in self.user_matrix.index:
                        n = int(self.hybrid_num.get()) // 2  # Half of total for collaborative

                        # Get average ratings and user's ratings
                        avg_ratings = self.user_matrix.mean()
                        user_ratings = self.user_matrix.loc[user_id]

                        # Find movies user hasn't rated
                        rated_movies = set(user_ratings[user_ratings > 0].index)
                        unrated_movies = avg_ratings[~avg_ratings.index.isin(rated_movies)]

                        # Get top unrated movies by average rating
                        top_movies = unrated_movies.sort_values(ascending=False).head(n).index.tolist()

                        # Store collaborative recommendations
                        for movie_id in top_movies:
                            movie_info = self.movies[self.movies['movieId'] == movie_id]
                            if not movie_info.empty:
                                movie = movie_info.iloc[0]
                                if movie['title'] not in collab_recs:
                                    collab_recs[movie['title']] = avg_ratings[movie_id]
                except:
                    continue  # Skip invalid user IDs

            # Display collaborative recommendations
            if collab_recs:
                self.hybrid_results.insert(tk.END, f"\nFrom {len(user_terms)} user(s): {', '.join(user_terms)}\n")
                self.hybrid_results.insert(tk.END, "-" * 30 + "\n")

                # Sort by average rating and display
                sorted_collab = sorted(collab_recs.items(), key=lambda x: x[1], reverse=True)
                start_num = len(recommendations) + 1  # Continue numbering

                for i, (title, avg_rating) in enumerate(sorted_collab[:int(self.hybrid_num.get()) // 2], start_num):
                    if title not in recommendations:
                        recommendations.append(title)
                        self.hybrid_results.insert(tk.END, f"{i}. {title}\n")
                        self.hybrid_results.insert(tk.END, f"   [Collaborative] Avg: {avg_rating:.2f}/5\n\n")

        # Update status bar
        self.status.config(text=f"Generated {len(recommendations)} hybrid recommendations")

    def search_movies(self):
        """SEARCH FUNCTION: Find movies by title (supports multiple search terms)"""
        if self.movies is None:
            return messagebox.showwarning("Warning", "Please load data first")

        term = self.search_input.get().strip()
        if not term:
            return messagebox.showwarning("Warning", "Please enter search term(s)")

        # Clear previous results
        self.search_results.delete('1.0', tk.END)
        search_terms = term.split()
        total_matches = 0

        # For each search term
        for search_term in search_terms:
            # Find movies containing the search term
            matches = self.movies[self.movies['title'].str.contains(search_term, case=False, na=False)]
            if len(matches) == 0:
                continue  # Skip if no matches

            total_matches += len(matches)

            # Display results for this search term
            self.search_results.insert(tk.END, f"Results for '{search_term}':\n")
            self.search_results.insert(tk.END, "-" * 30 + "\n")

            # Display each matching movie (limit to first 10)
            for _, row in matches.head(10).iterrows():
                self.search_results.insert(tk.END, f"• {row['title']}\n")
                self.search_results.insert(tk.END, f"  Rating: {self.get_movie_rating(row['movieId'])}\n")
                self.search_results.insert(tk.END, f"  Genres: {row['genres'][:50]}\n\n")

        # If no movies found, show message
        if total_matches == 0:
            self.search_results.insert(tk.END, f"No movies found for '{term}'\n")

        # Update status bar
        self.status.config(text=f"Found {total_matches} movies for {len(search_terms)} term(s)")

    def submit_feedback(self):
        """Save user feedback to CSV file"""
        movie = self.fb_movie.get().strip()
        if not movie:
            return messagebox.showwarning("Warning", "Please enter a movie title")

        # Get feedback data
        rating = self.fb_rating.get()
        comments = self.fb_comments.get('1.0', tk.END).strip()
        file_exists = os.path.exists(self.feedback_file)

        try:
            # Open feedback file in append mode
            with open(self.feedback_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Write header if file doesn't exist
                if not file_exists:
                    writer.writerow(['Timestamp', 'Movie', 'Rating', 'Comments'])

                # Write feedback data with timestamp
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    movie,
                    rating,
                    comments
                ])

            # Clear form after submission
            self.fb_movie.set("")
            self.fb_rating.set(3)
            self.fb_comments.delete('1.0', tk.END)

            # Reload feedback display and show success message
            self.load_feedback()
            messagebox.showinfo("Success", "Thank you for your feedback!")
            self.status.config(text="Feedback saved successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save feedback: {str(e)}")

    def load_feedback(self):
        """Load and display feedback history from CSV file"""
        self.fb_history.delete('1.0', tk.END)

        # Check if feedback file exists
        if not os.path.exists(self.feedback_file):
            self.fb_history.insert(tk.END, "No feedback submitted yet.")
            return

        try:
            # Read feedback file
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

                # Check if there's any feedback (excluding header)
                if len(rows) <= 1:
                    self.fb_history.insert(tk.END, "No feedback submitted yet.")
                    return

                # Display feedback statistics
                self.fb_history.insert(tk.END, f"Total Feedback: {len(rows) - 1}\n")
                self.fb_history.insert(tk.END, "=" * 40 + "\n\nRecent:\n")

                # Display most recent 5 feedback entries (reverse order = newest first)
                for row in reversed(rows[-5:]):
                    if len(row) >= 4:  # Ensure row has all required columns
                        self.fb_history.insert(tk.END, f"• {row[1]}\n")  # Movie title
                        self.fb_history.insert(tk.END, f"  Rating: {'★' * int(row[2])}\n")  # Star rating
                        if row[3]:  # Comments
                            self.fb_history.insert(tk.END, f"  Comment: {row[3][:50]}\n")
                        self.fb_history.insert(tk.END, "\n")

        except Exception as e:
            self.fb_history.insert(tk.END, f"Error: {str(e)}")


if __name__ == "__main__":
    # Create the main Tkinter window
    root = tk.Tk()

    # Create the MovieRecommender application
    app = MovieRecommender(root)

    # Start the Tkinter event loop (makes the window appear)
    root.mainloop()