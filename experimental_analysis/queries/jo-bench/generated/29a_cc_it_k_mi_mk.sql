SELECT * FROM keyword AS k, movie_keyword AS mk, movie_info AS mi, complete_cast AS cc, info_type AS it WHERE it.info = 'release dates' AND k.keyword = 'computer-animation' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;