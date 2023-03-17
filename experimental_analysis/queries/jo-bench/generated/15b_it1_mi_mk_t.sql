SELECT * FROM info_type AS it1, movie_info AS mi, title AS t, movie_keyword AS mk WHERE it1.info = 'release dates' AND mi.note LIKE '%internet%' AND mi.info LIKE 'USA:% 200%' AND t.production_year BETWEEN 2005 AND 2010 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;