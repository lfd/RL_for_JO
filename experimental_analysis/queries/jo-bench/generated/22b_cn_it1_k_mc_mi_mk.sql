SELECT * FROM movie_keyword AS mk, movie_companies AS mc, company_name AS cn, info_type AS it1, movie_info AS mi, keyword AS k WHERE cn.country_code != '[us]' AND it1.info = 'countries' AND k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi.info IN ('Germany', 'German', 'USA', 'American') AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND cn.id = mc.company_id AND mc.company_id = cn.id;