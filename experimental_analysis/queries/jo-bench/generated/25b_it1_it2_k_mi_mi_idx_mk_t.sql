SELECT * FROM keyword AS k, info_type AS it2, movie_keyword AS mk, title AS t, movie_info_idx AS mi_idx, movie_info AS mi, info_type AS it1 WHERE it1.info = 'genres' AND it2.info = 'votes' AND k.keyword IN ('murder', 'blood', 'gore', 'death', 'female-nudity') AND mi.info = 'Horror' AND t.production_year > 2010 AND t.title LIKE 'Vampire%' AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;