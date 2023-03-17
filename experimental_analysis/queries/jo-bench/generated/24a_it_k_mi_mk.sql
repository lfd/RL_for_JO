SELECT * FROM movie_info AS mi, movie_keyword AS mk, info_type AS it, keyword AS k WHERE it.info = 'release dates' AND k.keyword IN ('hero', 'martial-arts', 'hand-to-hand-combat') AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%201%' OR mi.info LIKE 'USA:%201%') AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;