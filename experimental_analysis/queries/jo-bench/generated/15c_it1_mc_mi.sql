SELECT * FROM info_type AS it1, movie_info AS mi, movie_companies AS mc WHERE it1.info = 'release dates' AND mi.note LIKE '%internet%' AND mi.info IS NOT NULL AND (mi.info LIKE 'USA:% 199%' OR mi.info LIKE 'USA:% 200%') AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;