SELECT * FROM movie_info AS mi, movie_keyword AS mk WHERE mi.note LIKE '%internet%' AND mi.info LIKE 'USA:% 200%' AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id;