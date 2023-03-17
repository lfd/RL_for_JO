SELECT * FROM movie_info AS mi, info_type AS it1, aka_title AS at, movie_keyword AS mk WHERE it1.info = 'release dates' AND mi.note LIKE '%internet%' AND mi.info LIKE 'USA:% 200%' AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = at.movie_id AND at.movie_id = mk.movie_id AND mi.movie_id = at.movie_id AND at.movie_id = mi.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;