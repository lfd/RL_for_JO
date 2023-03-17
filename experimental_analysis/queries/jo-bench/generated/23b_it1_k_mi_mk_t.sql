SELECT * FROM movie_keyword AS mk, movie_info AS mi, info_type AS it1, keyword AS k, title AS t WHERE it1.info = 'release dates' AND k.keyword IN ('nerd', 'loner', 'alienation', 'dignity') AND mi.note LIKE '%internet%' AND mi.info LIKE 'USA:% 200%' AND t.production_year > 2000 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;