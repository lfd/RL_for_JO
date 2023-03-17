SELECT * FROM info_type AS it1, keyword AS k, complete_cast AS cc, movie_keyword AS mk, movie_info AS mi, comp_cast_type AS cct1, comp_cast_type AS cct2 WHERE cct1.kind = 'crew' AND cct2.kind != 'complete+verified' AND it1.info = 'countries' AND k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND mi.info IN ('Sweden', 'Germany', 'Swedish', 'German') AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;