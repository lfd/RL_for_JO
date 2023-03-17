SELECT * FROM keyword AS k, movie_info AS mi, movie_keyword AS mk WHERE mi.note LIKE '%internet%' AND mi.info IS NOT NULL AND (mi.info LIKE 'USA:% 199%' OR mi.info LIKE 'USA:% 200%') AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;