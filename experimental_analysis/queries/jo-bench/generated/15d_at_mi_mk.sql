SELECT * FROM movie_info AS mi, aka_title AS at, movie_keyword AS mk WHERE mi.note LIKE '%internet%' AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = at.movie_id AND at.movie_id = mk.movie_id AND mi.movie_id = at.movie_id AND at.movie_id = mi.movie_id;